"""
power_trace.py
==============

Phase-resolved (time-varying) power trace for a SINGLE GPU kernel, reconstructed
from the EnergAIzer analytical predictor.

WHY THIS EXISTS
---------------
`estimator.lookup(...)` returns a single *average* power number per kernel
(power = energy / time). But the analytical model internally decomposes every
kernel into execution phases and tracks, for each phase, how busy each hardware
unit (DRAM, L2, shared memory, math/Tensor cores) is. It then collapses all of
that into one time-averaged number and throws the breakdown away.

This module does NOT throw it away. It intercepts the per-phase activity factors
and the fitted per-unit power coefficients, applies the power formula to each
phase *separately*, and lays the phases out on the timeline as a staircase:

      load (start, mem-bound, low power)
        -> compute (work, math-bound, high power)
          -> writeback (end, low power)            ... repeated per "wave"
            -> tail wave (some SMs idle, lower power)

IMPORTANT: this curve is MODELED, not MEASURED. Real per-kernel power cannot be
sampled (NVML samples every ~10-100 ms; a kernel lasts micro-to-milliseconds).
The staircase is a structural estimate whose area (integral of power x time)
equals exactly the energy the predictor already reports (energy-conserving).

USAGE
-----
    from gee import get_gee
    from power_trace import predict_power_trace, plot_power_trace

    estimator = get_gee(..., dvfs_aware=False)        # non-DVFS estimator
    query = {"batch":1,"dimM":4096,"dimN":4096,"dimK":4096,
             "precM":"bf16","precA":"bf16","useTensorCore":True}
    qtype = ("gemm","tc","bf16_bf16")

    trace = predict_power_trace(estimator, query, qtype, target_freq=900)
    plot_power_trace(trace, save_path="power_trace.png")
"""

import copy
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Phase layout. For a GEMM, every wave runs three phases:
#   start = prologue / global->shared loads (DRAM + L2 bound, no math)
#   work  = main MMA loop                   (math + shared-mem bound)
#   end   = epilogue / writeback            (DRAM + L2 bound, no math)
# "full_capacity" = the N-1 full waves; "last_wave" = the partial tail wave.
# -----------------------------------------------------------------------------
_PHASES = [
    # (human label, phase key, wave key, is_full_capacity)
    ("load",      "start", "full_capacity", True),
    ("compute",   "work",  "full_capacity", True),
    ("writeback", "end",   "full_capacity", True),
    ("tail load",      "start", "last_wave", False),
    ("tail compute",   "work",  "last_wave", False),
    ("tail writeback", "end",   "last_wave", False),
]
_UNITS = ["dram", "l2", "smem", "math"]


def _get(row, col, default=0.0):
    """Safe scalar read from a pandas row / dict (NaN -> default)."""
    try:
        v = row[col]
    except (KeyError, IndexError):
        return default
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
    except TypeError:
        pass
    return float(v)


def predict_power_trace(estimator, query, query_type, target_freq=900,
                        max_plot_waves=6, verbose=False):
    """
    Reconstruct a phase-resolved power trace for one GEMM kernel.

    Parameters
    ----------
    estimator : gee.gpu_energy_estimator.Gee
        Built with dvfs_aware=False (clean non-DVFS power model).
    query : dict
        {batch, dimM, dimN, dimK, precM, precA, useTensorCore}
    query_type : tuple
        e.g. ("gemm", "tc", "bf16_bf16")
    target_freq : int
        Core/SM clock in MHz used for the latency conversion.
    max_plot_waves : int
        Cap on how many full waves to draw individually (visual only; energy is
        preserved by scaling each drawn wave's duration).

    Returns
    -------
    dict with keys:
        segments     : list of (label, duration_ms, power_W) in execution order
        total_time_ms, avg_power_W, energy_J
        n_waves, flag, coeffs (per-unit power coeffs), modeled (bool)
    """
    from gee.optimization_utils import optimize  # lazy: needs PKG on sys.path

    est = estimator.gemm_estimator
    op = "gemm"

    # --- 1) Predict the concrete kernel (tiling/wave structure) for this shape
    q = copy.deepcopy(query)
    est.predict_kernel(q, op, predict_with_smaller_kernels=False, verbose=verbose)
    q["avg_freq"] = target_freq

    # --- 2) Run the analytical model -> per-phase t_* and activity columns
    qdf = pd.DataFrame([q], index=[0])
    est.analytical_model.model(qdf, train=False)
    q = dict(qdf.loc[0])

    # --- 3) Pull measured reference kernels of the same kernel type from the LUT.
    #        ignore_exact=True forces the regression path (so we always get a set
    #        to fit, even when an exact shape match exists in the LUT).
    flag, ref = est.get_references(
        q, op, condition_exact_match=True, condition_subset=est.fields,
        ignore_exact=True, min_entries=2, max_entries=20,
    )
    if ref is None or len(ref) < 2:
        return _flat_fallback(estimator, query, query_type, target_freq,
                              reason="not enough LUT references (flag=%s)" % flag)

    # --- 4) Fit latency coefficients (lambdas) on the references
    estimated_time, lambdas = est._predict_time(
        q, ref, target_freq, time_coeff_correction=[], verbose=verbose)
    if lambdas is None:
        return _flat_fallback(estimator, query, query_type, target_freq,
                              reason="arch-aware time fit failed (no lambdas)")

    # --- 5) Fit per-unit power coefficients on the references (non-DVFS form):
    #          power = p_dram*a_dram + p_l2*a_l2 + p_smem*a_smem
    #                + p_math*a_math + p_other*a_other + p_static
    ref = copy.deepcopy(ref)
    est._power_activity_factors(ref, lambdas, True, dvfs=False)
    z = ref[["a_dram", "a_l2", "a_smem", "a_math", "a_other"]].to_numpy()
    y = (1000.0 * ref["energy"].values / ref["time"].values)   # measured avg power
    p_dram, p_l2, p_smem, p_math, p_other, p_static = optimize(z, y)
    pcoeff = {"dram": p_dram, "l2": p_l2, "smem": p_smem, "math": p_math}

    # --- 6) Compute THIS query's per-phase activity factors and time weights
    qdf2 = pd.DataFrame([q], index=[0])
    est._power_activity_factors(qdf2, lambdas, False, dvfs=False)
    qrow = qdf2.loc[0]

    n_waves = int(round(_get(qrow, "n_waves", 1.0)))
    a_other = _get(qrow, "a_other", 0.0)
    T = float(estimated_time)          # total kernel time in ms

    # baseline floor present for the whole kernel
    static = float(p_static)

    # per-phase dynamic power and time-fraction weight
    def phase_power(phase, wave):
        p = 0.0
        for u in _UNITS:
            p += pcoeff[u] * _get(qrow, "a_%s_%s_%s" % (u, phase, wave), 0.0)
        return p + static

    # weight_* are fractions of total time (they sum to 1 - a_other)
    w = {(ph, wv): _get(qrow, "weight_%s_%s" % (ph, wv), 0.0)
         for _, ph, wv, _ in _PHASES}

    # --- 7) Build the staircase in execution order (energy-conserving) ---------
    segments = []
    n_full = max(n_waves - 1, 0)

    # full-capacity waves: optionally expand into individual waves for the plot
    if n_full > 0:
        draw = min(n_full, max_plot_waves)
        for wi in range(draw):
            for label, ph, wv, is_full in _PHASES[:3]:
                dur = (w[(ph, wv)] * T) / draw    # scale so total full-wave time kept
                pw = phase_power(ph, wv)
                tag = label if draw == 1 else "%s w%d" % (label, wi + 1)
                segments.append((tag, dur, pw))

    # the tail (last) wave -> usually lower power (some SMs idle)
    for label, ph, wv, is_full in _PHASES[3:]:
        dur = w[(ph, wv)] * T
        if dur <= 0:
            continue
        segments.append((label, dur, phase_power(ph, wv)))

    # constant-overhead remainder (keeps the integral exact)
    if a_other > 0:
        segments.append(("overhead", a_other * T, p_other + static))

    # drop zero-length slivers
    segments = [(l, d, p) for (l, d, p) in segments if d > 0]

    # --- 8) Report + energy-conservation check --------------------------------
    avg_power = estimator.lookup(copy.deepcopy(query), query_type,
                                 target_freq=target_freq, lookup_target="power")
    if isinstance(avg_power, pd.Series):
        avg_power = float(avg_power.values[0])
    energy_J = float(avg_power) * T / 1000.0
    trace_energy = sum(d * p for _, d, p in segments) / 1000.0

    if verbose:
        print("n_waves=%d  total_time=%.4f ms  flag=%s" % (n_waves, T, flag))
        print("power coeffs: dram=%.1f l2=%.1f smem=%.1f math=%.1f other=%.1f static=%.1f"
              % (p_dram, p_l2, p_smem, p_math, p_other, p_static))
        print("avg power (lookup) = %.2f W | trace-integrated = %.2f W"
              % (avg_power, trace_energy / T * 1000.0))

    return {
        "segments": segments,
        "total_time_ms": T,
        "avg_power_W": float(avg_power),
        "energy_J": energy_J,
        "trace_energy_J": trace_energy,
        "n_waves": n_waves,
        "flag": flag,
        "coeffs": {"dram": p_dram, "l2": p_l2, "smem": p_smem,
                   "math": p_math, "other": p_other, "static": p_static},
        "query": dict(query),
        "modeled": True,
    }


def demo_synthetic_trace(batch=1, M=4096, N=4096, K=4096, n_waves=4):
    """
    Self-contained illustrative trace (NO framework / LUT needed).

    Uses representative A100 bf16 Tensor-Core activity factors and power
    coefficients so the staircase shape can be drawn anywhere (e.g. to sanity
    check plotting). The REAL numbers come from `predict_power_trace`; this is
    only a shape illustration and is clearly labelled MODELED/SYNTHETIC.
    """
    # representative fitted power coefficients (W per unit of activity)
    p = {"dram": 85.0, "l2": 45.0, "smem": 35.0, "math": 165.0,
         "other": 25.0, "static": 55.0}

    # representative per-phase activity factors (fraction of phase time a unit is busy)
    # start = memory load (no math); work = compute (math hot); end = writeback
    act = {
        ("start", "full"): {"dram": 0.90, "l2": 0.70, "smem": 0.00, "math": 0.00},
        ("work",  "full"): {"dram": 0.12, "l2": 0.30, "smem": 0.60, "math": 0.95},
        ("end",   "full"): {"dram": 0.80, "l2": 0.55, "smem": 0.40, "math": 0.00},
        # tail wave: fewer busy SMs -> scale dynamic activity down
        ("start", "last"): {"dram": 0.55, "l2": 0.42, "smem": 0.00, "math": 0.00},
        ("work",  "last"): {"dram": 0.08, "l2": 0.18, "smem": 0.36, "math": 0.57},
        ("end",   "last"): {"dram": 0.48, "l2": 0.33, "smem": 0.24, "math": 0.00},
    }
    # per-wave phase durations (ms)
    dur = {"start": 0.020, "work": 0.110, "end": 0.015}

    def pw(phase, wave):
        a = act[(phase, wave)]
        return (p["dram"] * a["dram"] + p["l2"] * a["l2"]
                + p["smem"] * a["smem"] + p["math"] * a["math"] + p["static"])

    segments = []
    for wi in range(max(n_waves - 1, 0)):
        segments.append(("load w%d" % (wi + 1),      dur["start"], pw("start", "full")))
        segments.append(("compute w%d" % (wi + 1),   dur["work"],  pw("work", "full")))
        segments.append(("writeback w%d" % (wi + 1), dur["end"],   pw("end", "full")))
    segments.append(("tail load",      dur["start"] * 0.7, pw("start", "last")))
    segments.append(("tail compute",   dur["work"] * 0.6,  pw("work", "last")))
    segments.append(("tail writeback", dur["end"] * 0.7,   pw("end", "last")))
    segments.append(("overhead", 0.010, p["other"] + p["static"]))

    T = sum(d for _, d, _ in segments)
    energy = sum(d * q for _, d, q in segments) / 1000.0
    avg = energy / T * 1000.0
    return {
        "segments": segments, "total_time_ms": T, "avg_power_W": avg,
        "energy_J": energy, "trace_energy_J": energy, "n_waves": n_waves,
        "flag": "synthetic", "coeffs": p,
        "query": {"batch": batch, "dimM": M, "dimN": N, "dimK": K, "precM": "bf16"},
        "modeled": True, "synthetic": True,
    }


def _flat_fallback(estimator, query, query_type, target_freq, reason=""):
    """When phase decomposition is unavailable, return a single flat segment."""
    out = estimator.lookup(copy.deepcopy(query), query_type,
                           target_freq=target_freq, lookup_target="all")
    t, p, e = out[0], out[1], out[2]
    if isinstance(t, pd.Series):
        t = float(t.values[0])
    if isinstance(e, pd.Series):
        e = float(e.values[0])
    p = float(p) if p not in (None, -1) else (e / t * 1000.0 if t else 0.0)
    return {
        "segments": [("kernel (avg)", float(t), p)],
        "total_time_ms": float(t),
        "avg_power_W": p,
        "energy_J": float(e),
        "trace_energy_J": float(e),
        "n_waves": 1,
        "flag": "flat",
        "coeffs": {},
        "query": dict(query),
        "modeled": False,
        "fallback_reason": reason,
    }


def trace_to_curve(segments):
    """Turn segments -> (time_axis_ms, power_axis_W) step arrays for plotting."""
    xs, ys = [0.0], []
    t = 0.0
    for _, dur, pw in segments:
        ys.append(pw)
        t += dur
        xs.append(t)
    return np.array(xs), np.array(ys)


def plot_power_trace(trace, save_path=None, title=None, show=False):
    """Draw the phase-resolved power staircase. Returns the matplotlib figure."""
    import matplotlib
    if save_path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segs = trace["segments"]
    xs, ys = trace_to_curve(segs)

    fig, ax = plt.subplots(figsize=(10, 4.8))

    # staircase
    ax.step(xs, np.append(ys, ys[-1]), where="post", color="#1f3b73", lw=2.0,
            zorder=3)
    # shade phases by kind (compute hot, memory cool, overhead grey)
    t = 0.0
    for label, dur, pw in segs:
        kind = label.split()[0]
        color = {"load": "#9ecae1", "tail": "#9ecae1", "writeback": "#9ecae1",
                 "compute": "#fc9272", "overhead": "#d9d9d9"}.get(kind, "#9ecae1")
        if "compute" in label:
            color = "#fc9272"
        if label == "overhead":
            color = "#d9d9d9"
        ax.axvspan(t, t + dur, color=color, alpha=0.35, zorder=1)
        t += dur

    # average-power reference line
    avg = trace["avg_power_W"]
    ax.axhline(avg, ls="--", color="black", lw=1.2, zorder=2,
               label="average power = %.1f W" % avg)

    ax.set_xlabel("time (ms)")
    ax.set_ylabel("power (W)")
    ax.set_xlim(0, xs[-1])
    ax.set_ylim(0, max(ys) * 1.18)
    q = trace.get("query", {})
    if title is None:
        title = ("Phase-resolved power trace (MODELED)  |  "
                 "GEMM B=%s M=%s N=%s K=%s %s  |  %d waves, %.3f ms, %.4f J"
                 % (q.get("batch"), q.get("dimM"), q.get("dimN"), q.get("dimK"),
                    q.get("precM", ""), trace["n_waves"], trace["total_time_ms"],
                    trace["energy_J"]))
    ax.set_title(title, fontsize=10)

    # legend for the phase colors
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#9ecae1", alpha=0.5, label="memory phase (load / writeback)"),
        Patch(facecolor="#fc9272", alpha=0.5, label="compute phase (math / Tensor cores)"),
        Patch(facecolor="#d9d9d9", alpha=0.6, label="overhead"),
    ]
    leg1 = ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print("saved:", save_path)
    if show:
        plt.show()
    return fig


def print_trace(trace):
    """Human-readable segment table."""
    print("\n%-18s %12s %12s" % ("phase", "duration(ms)", "power(W)"))
    print("-" * 44)
    t = 0.0
    for label, dur, pw in trace["segments"]:
        print("%-18s %12.5f %12.2f" % (label, dur, pw))
        t += dur
    print("-" * 44)
    print("%-18s %12.5f %12.2f  (avg)" % ("TOTAL", t, trace["avg_power_W"]))
    print("energy: lookup=%.5f J   trace-integral=%.5f J   (match = energy-conserving)"
          % (trace["energy_J"], trace["trace_energy_J"]))
    if not trace.get("modeled", True):
        print("NOTE: fell back to a flat segment (%s)"
              % trace.get("fallback_reason", ""))
