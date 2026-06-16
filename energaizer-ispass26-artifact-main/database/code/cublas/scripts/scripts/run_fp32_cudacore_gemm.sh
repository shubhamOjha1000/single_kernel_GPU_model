#!/bin/bash

# Get user's python and ncu bin commands and sudo password
if [ $# -lt 5 ] || [ $# -gt 6 ]; then
    echo "Usage: $0 <PYTHON_BIN> <NCU_BIN> <SUDO_PWD> <GPU_ID> <USER_ID>"
    exit 1
fi

# Get arguments
PYTHON_BIN="$1"
NCU_BIN="$2"
SUDO_PWD="$3"
CUDA_DEVICE="$4"
USER_ID="$5"

mkdir -p ./data

# 1. Create bash script and run
mkdir -p ./data/raw/fp32_cudacore/nvml
mkdir -p ./data/raw/fp32_cudacore/ncu
cd scripts
python3 generate_bash_script.py --workload_csv_path workloads/fp32_cudacore_gemm_workloads.csv --bash_save_path run_fp32_cudacore_gemm_collection.sh --profile_nvml --profile_ncu --nvml_save_path data/raw/fp32_cudacore/nvml --ncu_save_path data/raw/fp32_cudacore/ncu --transpose_options nn --cuda_device $CUDA_DEVICE --ncu_bin_path $NCU_BIN --lock_gpu_clock --gpu_clock_freq 900
cd ..
bash scripts/run_fp32_cudacore_gemm_collection.sh $SUDO_PWD
echo $SUDO_PWD | sudo -S chown $USER_ID data/raw/fp32_cudacore/ncu/spsA0p0_spsB0p0/*

# 2. Generate final database csv files
mkdir -p ./data/parsed/fp32_cudacore
cd scripts
python3 generate_lut.py --save_to ../data/parsed/fp32_cudacore --nvml_path ../data/raw/fp32_cudacore/nvml/spsA0p0_spsB0p0/ --enable_ncu --ncu_path ../data/raw/fp32_cudacore/ncu/spsA0p0_spsB0p0/
cd ..
