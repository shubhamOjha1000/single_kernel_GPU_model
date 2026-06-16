# Collecting Test Cases for EnergAIzer

This code provides scripts to collect test cases, including 1) input files for end-to-end estimation, 2) the groundtruth measurement of latency and energy consumption, and 3) optional Nsight Compute profiling results for deeper analysis/debugging. 

## Code Structure

### 1. Run workload

`run.py` provides a Python script with options to 1) generate traces of workloads (i.e., the operator types and tensor shapes), 2) run measurements for the groundtruth latency and energy consumption, and 3) run Nsight Compute (NCU) profiling. 

#### 1.1 Specifying the models to run

Currently, `run.py` accepts these models from HuggingFace/PyTorch:

- Module-level: OPTDecoder (a single decoder layer from [HF OPT definition](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L184))
- Language models from HF: OPT, BERT, GPT2, Qwen2
- Vision models from HF/PyTorch: ViT, MobileViT, ResNet

To run the code for any among these models, specify `--model_type` (i.e., whether it's a module-level, a language model, or a vision model), `--model` (i.e., the model name like OPT, BERT, etc.), and `--config_folder`, indicating the path to the folder that contains the HF-style configuration files for HF models. 
Note that `run.py` will iterate through all configuration files in the `--config_folder` (e.g., 6 configurations in [opt-all](workload_config/opt-all/)).

#### 1.2 Specifying the attention backend and torch.compile options

HF models can use different backends for self-attention operations. 
`run.py` currently supports two options: 1) `eager`, which is the vanilla baseline attention defined as a sequence of tensor operations in PyTorch (e.g., see [implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L74)), and 2) `sdpa`, which calls PyTorch's [scaled-dot-product-attention](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html) and typically resolves to FlashAttention when tensors are in half-precision floating-point. 
This backend option can be specified with `--attn_backend`. 

Additionally, if you want to use `torch.compile` that can improve the effiency by fusing pointwise operators with the preceding operators, you can use `--compile` flag. 

#### 1.3 Specifying the workload input size (batch, sequence length)

`run.py` allows you to sweep across different tensor precisions (`--precision`), batch sizes (`--batch`), sequence lengths (`--seqlen`), and prefill/decode settings (`--mode`). 

#### 1.4 Options for tracing, measuring, and profiling

These are GPU related options:
- Please set `--sudo_pwd` with your sudo password if you want to either 1) set the GPU frequency explicitly (i.e., using `--gpu_clock_freq_sweep` flag), or 2) run NCU profiling. 
- Please set `--python_bin_path` with a valid binary for executing Python. It can be as simple as `python` or a specific path to your virtual environment like `~/miniconda3/envs/[YOUR_ENV_NAME]/bin/python`. 
- Please set `--cuda_device` with the GPU ID if you are working in a multi-GPU server. 
- If you want to set the GPU frequency explicitly, set the `--gpu_clock_freq_sweep` flag. This will make the script to sweep across a range of operating frequencies, defined by [`--gpu_min_freq`, `--gpu_max_freq`], with a step size of `--gpu_freq_step`. You can set the min and max frequency to be equal if you want to lock to a single frequency. 

To trace the workloads, 
- Set the `--run_trace` flag, and specify the path you want to save the tracing results with `--trace_save_to`. 

Inside the script, this will run [torchlens](https://github.com/johnmarktaylor91/torchlens) to obtain the operators inside the workload. 
You don't need a GPU to run tracing, as we only need the operator types and tensor shapes. 
However, since torchlens needs one backward pass to extract the operators, running tracing on a CPU often runs out of memory. 
You can run this tracing for small batches/sequences, then extrapolate the tensor shapes for larger batches/sequences.
Alternatively, if you have a CPU/GPU with a large memory capacity, you can directly run this extraction on that machine. 

To measure the groundtruth latency and energy,
- Set the `--run_nvml` flag, and specify the path you want to save the measurement results with `--nvml_save_to`. 
- If you want to record the operating frequency throughout the measurement, set the `--nvml_poll_clock` flag. 
- If you want to change the NVML update period, use the `--nvml_update_period` flag. 

To run NCU profiling, 
- Set the `--run_ncu` flag, and specify the path you want to save the NCU logs with `--ncu_save_to`. 
- Specify the binary path for NCU with `--ncu_bin_path`. 

### 2. Parse files 

After `run.py` finishes, please parse the generated files using `parse_trace.py` and `parse_result.py`. 

#### 2.1. Parse traces

`parse_trace.py` will convert the torchlens output files generated by `run.py` into the input format required by EnergAIzer. 
The options are:

- `--trace_path`: the path to the folder containing the torchlens results, which is `--trace_save_to` that was used for `run.py`
- `--parsed_save_to`: the path where you want to save the parsed results
- `--fusion`: set this flag if you are using `--compile` option when running `run.py`


#### 2.2. Parse results

`parse_result.py` will summarize the latency and energy for each workload from the NVML results generated by `run.py`. 
The options are:

- `--path_to_folder`: the path to the folder containing the NVML results, which is `--nvml_save_to` used for `run.py`
- `--save_to`: the path to the folder where you want to save the summarized result

## Prepared Scripts

We provide scripts that run and parse the workloads in [scripts/](scripts/). 
You can change the options defined at the beginning of each script. 
For example, inside [`run_opt.sh`](scripts/run_opt.sh), you can define:
```bash
# Define these paths
PYTHON_BIN="~/miniconda3/envs/pytorch/bin/python3" # Python binary, --python_bin_path
NCU_BIN="/usr/local/cuda/bin/ncu"                  # NCU binary, --ncu_bin_path

# Define these options
CUDA_DEVICE=1                                      # GPU ID, --cuda_device

CLOCK_SWEEP=1                                      # Whether to set --gpu_clock_freq_sweep
CLOCK_MIN=900                                      # --gpu_min_freq
CLOCK_MAX=900                                      # --gpu_max_freq

NVML=true                                          # --run_nvml
NCU=true                                           # --run_ncu
TRACE=true                                         # --run_trace
COMPILE=false                                      # --compile

MODEL_TYPE="LanguageModel"                         # --model_type
MODEL="OPTModel"                                   # --model
CONFIG="workload_config/opt-subset/"               # --config_folder
PREC="bf16"                                        # --precision
BATCH="1 2 4 8"                                    # --batch
SEQLEN="128 512 1024 2048"                         # --seqlen
MODE="prefill"                                     # --mode
BACKEND="sdpa"                                     # --attn_backend
```

Then, when executing, you can specify the paths and sudo password:
```bash
bash run_opt.sh [USER_ID] [SUDO_PWD] [FOLDER]
```
The parsed results (NVML/NCU) will be located at `save/[FOLDER]_parsed`.
The parsed traces will be located at `save/[FOLDER]_trace_parsed`. 

## Custom Models

If you want to prepare the EnergAIzer inputs and measure the groundtruth latency/energy for AI workloads not currently specified by this code, consider these two options.

### Option 1: HF/PyTorch models -> add them to `run_model.py`

If your workload is another HF or PyTorch model, you can extend `run_model.py` to support your model. 
`run_model.py` gets called by `run.py`, and it imports the model definitions from HF's `transformers` and PyTorch's `torchvision.models`. 
You can import your model definition and add it to `get_model` function, then use this code in the same manner. 
However, make sure to sanity check if the generated traces/inputs reflect your model correctly. 

### Option 2: If you prefer to write your own scripts

When the simple extension in the above does not work for your workload for tracing, you might want to prepare it with your own scripts. 
EnergAIzer's input format (e.g., JSON files in [`../data/workloads/all`](../data/workloads/all/)) is a Python list of operators inside the workload. 
Each element represents a tensor operation, described as a tuple. 
Please see the examples below:

- GEMM operations: `[{"batch": 1, "dimM": 1024, "dimN": 128, "dimK": 1024, "precM": "bf16", "precA": "bf16", "useTensorCore": true}, ["gemm", "tc", "bf16_bf16"]]`
  - First item is a dictionary specifying the tensor size (batch, dimM, dimN, dimK), precision (precM, precA), whether it uses Tensor cores (useTensorCore). 
  - Second item states the operator type. In this example, it says it's a GEMM operation, using Tensor cores, and the precisions for both the operand and the result are in `bf16`. 
- Nonlinear reduction operations: `[{"batch": 2048, "dim": 128, "prec": "bf16"}, ["softmax", "bf16"]]`
  - First item is specifying the tensor size (batch, dim) and the precision (prec).
  - Second item states the operator type (e.g., softmax in `bf16` precision).
- Elementwise operations: `[{"dim": 262144, "op": "pointwise_add", "prec": "bf16"}, ["elementwise"]]`
  - First item is specifying the tensor size (dim, which is a flattened tensor's element count), precision (prec), and the specific operation (op). In this example, the operation is a pointwise addition between two tensors. 
  - In the current version, EnergAIzer supports pointwise addition (`pointwise_add`), pointwise multiplication (`pointwise_mul`), scalar addition that adds a constant to a tensor (`scalar_add`), scalar multiplication (`scalar_mul`), ReLU (`relu`), GeLU (`gelu`), Tanh (`tanh`), SiLu (`silu`), and Sigmoid (`sigmoid`). If you have elementwise operation that does not exactly fall into these categories, you can approximate them as either `unspecified_tensor` (when the operand involves two tensors of the same shape) or `unspecified_scalar` (when the operand involves a single tensor). 

You can prepare inputs to EnergAIzer following this format for end-to-end estimation.
You can also call EnergAIzer's single kernel estimation function for each of the operation in your model instead of using the end-to-end scripts, then aggregate the latency and energy estimation. 