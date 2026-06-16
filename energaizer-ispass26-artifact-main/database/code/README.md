# Database Collection for EnergAIzer

This code provides scripts for offline database collection required for EnergAIzer. 

## Setup

EnergAIzer collects database for the following kernel types:

- Generalized matrix multiplication (GEMM) using cuBLAS
- Nonlinear reduction functions (e.g., Softmax, Layer Normalization) from PyTorch
- FlashAttention kernels from PyTorch
- Convolution kernels from PyTorch assuming the cuDNN backend
- Elementwise operations (e.g., activation functions, misc. tensor/scalar operations) from PyTorch
- (Experimental) Collective communication kernels from NCCL

Accordingly, this code provides data collection scripts for three software libraries: cuBLAS, PyTorch, and NCCL. 
Please make sure you have NVIDIA GPUs (CUDA 12.4+) and Nsight Compute (requires sudo).

### cuBLAS

We adapt the [tensor core benchmarking code](https://github.com/hibagus/CUDA_Bench/tree/main) by adding [NVML monitor](https://github.com/mnicely/nvml_examples/tree/master) to record latency and power consumption.
The code has been built and tested for this environment:

- CUDA 12.4
- Ubuntu 22.04.4 LTS
- cpp 11.4.0
- cmake 3.14.3

To setup the code, please run the script:
```bash
bash misc/cublas.sh
```

### PyTorch

Please make sure you created a conda virtual environment `energaizer_env` and activate it. 

If you don't have anaconda or miniconda installed, please use the script `misc/install_conda.sh`. 
If you didn't create a virtual environment, please use `conda env create -f misc/conda_env.yml`.

### NCCL

We adapt the [NCCL Test](https://github.com/NVIDIA/nccl-tests/tree/8dfeab9eb9bdfdf13503e71e1f33e7f8a208b540) repository by adding NVML monitor, similar to cuBLAS benchmarking. 
The code has been built and tested for CUDA 12.4. 
To setup the code, please follow the instructions in the [NCCL Test](https://github.com/NVIDIA/nccl-tests/tree/8dfeab9eb9bdfdf13503e71e1f33e7f8a208b540) repository. 

## Scripts

All scripts follow a similar structure: 
(1) generate and execute a bash script that runs data collection, and
(2) parses collected information (power, latency, and kernel information) into a database csv file.
Note that final database csv files can be used to fit EnergAIzer (specify them in the yaml files when configuring EnergAIzer). 

Make sure to check the scripts and set the save path to your desired location. 

### 1. GEMM (cuBLAS)

A pre-defined script that collects half-precision tensor core GEMM kernels at 900 MHz can be launched with:
```bash
cd cublas
bash scripts/scripts/run_bf16_tensorcore_gemm.sh [PYTHON_BIN] [NCU_BIN] [SUDO_PWD] [GPU_ID] [USER_ID]
```
For example, `PYTHON_BIN` can be `~/miniconda3/envs/energaizer_env/bin/python3` and `NCU_BIN` can be `/usr/local/cuda/bin/ncu`.
Please specifiy `GPU_ID` even if there is a single GPU in your machine, by simply setting it to 0. 
Provide your `USER_ID` and `SUDO_PWD` to use Nsight Compute. 

You can modify the script:

- Workloads to be benchmarked: modify or add new csv files in `cublas/scripts/workloads`; specify the GEMM problem shape (batch, M/N/K) and the number of iterations for benchmarking (ensure at least a few seconds of execution for stable power readings for NVML)
- Change GPU's operating frequency: change `--gpu_clock_freq` option when generating the bash script
- Not collecting Nsight Compute: remove `--profile_ncu` option when generating the bash script
- Make sure to the change the folder and result file names

### 2. Nonlinear Reduction (PyTorch)

An example script can be launched as:
```bash
cd pytorch/nn
bash scripts/run_nonlinear.sh [PYTHON_BIN] [NCU_BIN] [SUDO_PWD] [GPU_ID] [USER_ID]
```

Similar as above, you can modify the workloads to be benchmarked by adding/modifying csv files in `pytorch/nn/workloads`. 

### 3. Activations and Other Elementwise (PyTorch)

An example script to collect activation functions (e.g., ReLU, GELU) can be launched as:
```bash
cd pytorch/nn
bash scripts/run_activation.sh [PYTHON_BIN] [NCU_BIN] [SUDO_PWD] [GPU_ID] [USER_ID]
```

For other elementwise operations, such as pointwise additions/multiplications, an example script is:
```bash
cd pytorch/elementwise
bash scripts/run.sh [PYTHON_BIN] [NCU_BIN] [SUDO_PWD] [GPU_ID] [USER_ID]
```

Please concatenate the final database files into one csv file for elementwise operations. 

### 4. Convolution (PyTorch/cuDNN)

An example script to collect PyTorch's 2D convolutions is:
```bash
cd pytorch/nn
bash scripts/run_conv.sh [PYTHON_BIN] [NCU_BIN] [SUDO_PWD] [GPU_ID] [USER_ID]
```

### 5. FlashAttention (PyTorch)

An example script to collect PyTorch's FlashAttention is:
```bash
cd pytorch/flashattn
bash scripts/run.sh [PYTHON_BIN] [NCU_BIN] [SUDO_PWD] [GPU_ID] [USER_ID]
```

### 6. Collective Comunication (NCCL)

An example script to collect collective communication kernels is:
```bash
cd nccl
bash scripts/scripts/run.sh [SUDO_PWD] [NUM_GPUs]
```
Note that the benchmarking script will use GPUs 0, ..., (NUM_GPUs-1).
Also, Nsight Compute profiling is **not** supported for NCCL kernels in the current codebase. 

Please check `nccl/src` for how the workload sizes (in bytes) are defined for send and receive operations. 