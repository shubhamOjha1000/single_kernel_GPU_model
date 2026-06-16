#!/bin/bash

# Get user's python and ncu bin commands and sudo password
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <SUDO_PWD> <NUM_GPUS>"
    echo "It will use GPUs 0, ...., NUM_GPUS-1 for benchmarking"
    exit 1
fi

# Get arguments
SUDO_PWD="$1"
NUM_GPU="$2"

mkdir -p ./data

# 1. Create bash script and run
mkdir -p ./data/raw
cd scripts
python3 generate_bash_script.py --workload_csv_path workloads/nccl_workloads.csv --bash_save_path run_nccl_collection.sh --nccl_save_path data/raw --nccl_save_subpath gpu$NUM_GPU --n_gpus $NUM_GPU
cd ..
CUDA_VISIBLE_DEVICES=$(seq 0 $((NUM_GPU-1)) | paste -sd ',') bash scripts/run_nccl_collection.sh

# 2. Generate final database csv files
mkdir -p ./data/parsed/
cd scripts
python3 parse_folder.py --path_to_folder ../data/raw/gpu${NUM_GPU}_0 --save_to ../data/parsed/gpu${NUM_GPU}_database.csv --n_gpus $NUM_GPU
cd ..
