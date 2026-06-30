# single_kernel_GPU_model

GPU **latency / power / energy** prediction for AI workloads, built on the
[EnergAIzer (ISPASS'26)](energaizer-ispass26-artifact-main/README.md) analytical framework
(vendored in [`energaizer-ispass26-artifact-main/`](energaizer-ispass26-artifact-main/)).

This repo adds a **single-kernel, time-varying power trace** on top of EnergAIzer.

---

## What's here

| File | Purpose |
|------|---------|
| [`power_trace.py`](power_trace.py) | Reconstructs a **phase-resolved (time-varying) power curve** for one GPU kernel from the EnergAIzer predictor. |
| [`Single_Kernel_Power_Trace_Colab.ipynb`](Single_Kernel_Power_Trace_Colab.ipynb) | **CPU Colab notebook**: clones this repo, downloads the LUT database, and draws the power-vs-time staircase. No GPU needed. |
| [`EnergAIzer_Colab_Demo.ipynb`](EnergAIzer_Colab_Demo.ipynb) | The original end-to-end EnergAIzer demo (single kernel, whole-model energy, DVFS, design-space exploration). |
| [`energaizer-ispass26-artifact-main/`](energaizer-ispass26-artifact-main/) | The vendored EnergAIzer framework (`gee` package, configs, LUT loader). |

---

## The idea: average power -> a power *curve*

`estimator.lookup(...)` returns **one average power number** per kernel (`power = energy / time`).
But the analytical model internally decomposes every kernel into execution **phases** and tracks how
busy each hardware unit (DRAM, L2, shared memory, math / Tensor cores) is in each phase — then
**averages it all away**.

`power_trace.py` keeps that breakdown. It applies the fitted per-unit power coefficients to **each
phase separately** and lays them on the timeline as a **staircase**:

```
 load (mem-bound, low)  ->  compute (math-bound, high)  ->  writeback (low)     [repeated per wave]
   ... -> tail wave (some SMs idle -> lower power)
```

### MODELED, not MEASURED

Real per-kernel power **cannot** be sampled: NVML/`nvidia-smi` samples every **~10–100 ms**, while a
single kernel runs in **micro-to-milliseconds**. Capturing a true within-kernel curve needs external
**shunt/DAQ probes at kHz–MHz**. This staircase is the predictor's **structural estimate** — but it is
**energy-conserving**: the area under the curve (∑ power × duration) equals **exactly** the energy
`lookup()` reports.

---

## Quick start (Colab, CPU)

1. Open [`Single_Kernel_Power_Trace_Colab.ipynb`](Single_Kernel_Power_Trace_Colab.ipynb) in Google Colab
   (Runtime = CPU is fine).
2. Run the cells top to bottom. They clone the repo, install CPU deps, download the LUT, build the
   estimator, and plot the trace.

## Quick start (local Python)

```python
from gee import get_gee                       # from energaizer-ispass26-artifact-main (on sys.path)
from power_trace import predict_power_trace, print_trace, plot_power_trace

estimator = get_gee(
    gpu_yaml_path="energaizer-ispass26-artifact-main/config/gpu/yz8.yaml",   # A100-40GB-PCIE
    lut_yaml_path="energaizer-ispass26-artifact-main/experiments_endtoend/exp_config/a100_lut_config.yaml",
    dvfs_aware=False,                          # clean non-DVFS power model
    lut_folder_abs_path="energaizer-ispass26-artifact-main/database/data",
)

query = {"batch": 1, "dimM": 4096, "dimN": 4096, "dimK": 4096,
         "precM": "bf16", "precA": "bf16", "useTensorCore": True}
query_type = ("gemm", "tc", "bf16_bf16")

trace = predict_power_trace(estimator, query, query_type, target_freq=900)
print_trace(trace)
plot_power_trace(trace, save_path="power_trace.png")
```

`predict_power_trace(...)` returns a dict with:
`segments` (list of `(label, duration_ms, power_W)`), `total_time_ms`, `avg_power_W`, `energy_J`,
`trace_energy_J` (should equal `energy_J`), `n_waves`, and the fitted per-unit `coeffs`.

> Need just the concept without the LUT? `power_trace.demo_synthetic_trace()` returns an illustrative
> (clearly **synthetic**) trace using representative A100 bf16 activity factors.

---

## How the LUT is obtained

The notebook downloads the pre-collected database (~hundreds of MB) from Google Drive
([file](https://drive.google.com/file/d/1krvqRFDnaqrJUT06V2psIua0wQr6ETAE/view)) and flattens the CSVs
into `energaizer-ispass26-artifact-main/database/data/`. See the
[EnergAIzer README](energaizer-ispass26-artifact-main/README.md) for the full framework setup.

## Credit

Built on **EnergAIzer: Fast and Accurate GPU Power Estimation Framework for AI Workloads** (ISPASS'26).
See [`energaizer-ispass26-artifact-main/LICENSE`](energaizer-ispass26-artifact-main/LICENSE).
