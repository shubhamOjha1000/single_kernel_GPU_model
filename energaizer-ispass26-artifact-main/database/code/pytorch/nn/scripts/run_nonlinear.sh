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

mkdir -p "./data"

# 1. Generate bash script for data collection
mkdir -p "./data/raw/nonlinear"
python3 generate_bash_script_nn.py --profile_nvml --nvml_save_path data/raw/nonlinear --workload_csv_path workloads/nonlinear_workload.csv --bash_save_path data/run_data_collection_nonlinear.sh --profile_ncu --ncu_save_path data/raw/nonlinear --ncu_bin_path $NCU_BIN --python_bin_path $PYTHON_BIN --cuda_device $CUDA_DEVICE --lock_gpu_clock --gpu_clock_freq 900 --sudo_password $SUDO_PWD
bash data/run_data_collection_nonlinear.sh
echo $SUDO_PWD | sudo -S chown $USER_ID data/raw/nonlinear/*

# 2. Parse results
mkdir -p "./data/parsed/nonlinear"
python3 ../parse_folder.py --path_to_folder data/raw/nonlinear --save_to data/parsed/nonlinear

# 3. Generate database csv file
python3 generate_lut_nonlinear.py --result_folder data/parsed/nonlinear --operation_name softmax --save_to data --save_prefix softmax --save_suffix freq900
python3 generate_lut_nonlinear.py --result_folder data/parsed/nonlinear --operation_name layernorm --save_to data --save_prefix layernorm --save_suffix freq900
