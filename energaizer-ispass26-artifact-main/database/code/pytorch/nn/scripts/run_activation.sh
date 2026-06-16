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
mkdir -p "./data/raw/activation"
python3 generate_bash_script_nn.py --profile_nvml --nvml_save_path data/raw/activation --workload_csv_path workloads/activation_workload.csv --bash_save_path data/run_data_collection_activation.sh --profile_ncu --ncu_save_path data/raw/activation --ncu_bin_path $NCU_BIN --python_bin_path $PYTHON_BIN --cuda_device $CUDA_DEVICE --lock_gpu_clock --gpu_clock_freq 900 --sudo_password $SUDO_PWD
bash data/run_data_collection_activation.sh
echo $SUDO_PWD | sudo -S chown $USER_ID data/raw/activation/*

# 2. Parse results
mkdir -p "./data/parsed/activation"
python3 ../parse_folder.py --path_to_folder data/raw/activation --save_to data/parsed/activation

# 3. Generate database csv file
python3 generate_lut_activation.py --result_folder data/parsed/activation --save_to data/activation_database_freq900.csv