import os
import shutil

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import re
import math
import yaml

import time
import copy
import json
from tqdm import tqdm

import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sdpa_result_folder', type=str, required=True, help='path to the folder with NVML/NCU results')
    parser.add_argument('--profile_ncu', default=False, action='store_true')
    parser.add_argument('--transfer_ncu', default=False, action='store_true')
    parser.add_argument('--transfer_ncu_files', nargs='*', default=None)
    parser.add_argument('--save_to', type=str, required=True)

    args = parser.parse_args()

    sdpa_result_folder = args.sdpa_result_folder
    ncu_profile = args.profile_ncu
    ncu_transfer = args.transfer_ncu
    ncu_transfer_from = args.transfer_ncu_files
    save_to = args.save_to

    df_nvml = pd.read_csv(os.path.join(sdpa_result_folder, 'nvml', 'nvml_parsed.csv'))
    df_nvml[['batch', 'n_head', 'q_seq_len', 'seq_len', 'head_dim']] = -1
    df_nvml['prec'] = None
    df_nvml['kernel_name'] = None
    df_nvml['block_size'] = None
    df_nvml['grid_size'] = None
    df_nvml['max_concurrent_block'] = -1
    df_nvml['is_flash'] = False
    df_nvml['precM'] = None
    df_nvml['precA'] = None
    df_nvml['useTensorCore'] = False

    df_nvml['found_ncu'] = False

    if ncu_transfer:
        df_ncu = None
        for p in ncu_transfer_from:
            _df = pd.read_csv(p)
            if 'q_seq_len' not in _df.keys():
                _df['q_seq_len'] = _df['seq_len']
            if df_ncu is None:
                df_ncu = _df
            else:
                df_ncu = pd.concat([df_ncu, _df], ignore_index=True)

    for idx, row in df_nvml.iterrows():
        workload = row['workload']

        if 'flash_attention' in workload:

            try:
                workload_shape = re.findall('_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_', workload)[0][1:-1].split('_')
                batch = int(workload_shape[0])
                n_head = int(workload_shape[1])
                q_seq_len = int(workload_shape[2])
                seq_len = int(workload_shape[3])
                head_dim = int(workload_shape[4])
                
            except:
                workload_shape = re.findall('_[0-9]+_[0-9]+_[0-9]+_[0-9]+_', workload)[0][1:-1].split('_')
                batch = int(workload_shape[0])
                n_head = int(workload_shape[1])
                seq_len = int(workload_shape[2])
                head_dim = int(workload_shape[3])
                q_seq_len = seq_len

            df_nvml.loc[idx, 'seq_len'] = seq_len
            df_nvml.loc[idx, 'batch'] = batch
            df_nvml.loc[idx, 'n_head'] = n_head
            df_nvml.loc[idx, 'head_dim'] = head_dim
            df_nvml.loc[idx, 'precM'] = 'bf16'
            df_nvml.loc[idx, 'precA'] = 'bf16'
            df_nvml.loc[idx, 'useTensorCore'] = True
            df_nvml.loc[idx, 'is_flash'] = True
            df_nvml.loc[idx, 'q_seq_len'] = q_seq_len

            if ncu_profile:
                workload_name_before_iter = workload.split('_iter')[0]

                try:
                    ncu_file_name = 'ncu_' + workload_name_before_iter + '.csv'
                    ncu_df = pd.read_csv(os.path.join(sdpa_result_folder, 'ncu', ncu_file_name))
                except:
                    ncu_file_name = 'parsed_ncu_' + workload_name_before_iter + '.csv'
                    ncu_df = pd.read_csv(os.path.join(sdpa_result_folder, 'ncu', ncu_file_name))
                ncu_df['max_concurrent_block'] = ncu_df.apply(lambda row: min(row['Block Limit SM'],row['Block Limit Registers'],row['Block Limit Shared Mem'],row['Block Limit Warps']), axis=1)

                flash_attn_row = ncu_df.loc[ncu_df['kernel_name'].str.contains('flash')]
                flash_attn_kernel_name = flash_attn_row['kernel_name'].values[0]
                block_size = flash_attn_row['block_size'].values[0]
                grid_size = flash_attn_row['grid_size'].values[0]
                max_concurrent_block = flash_attn_row['max_concurrent_block'].values[0]
                
                df_nvml.loc[idx, 'kernel_name'] = flash_attn_kernel_name
                df_nvml.loc[idx, 'block_size'] = block_size
                df_nvml.loc[idx, 'grid_size'] = grid_size
                df_nvml.loc[idx, 'max_concurrent_block'] = max_concurrent_block

            if ncu_transfer:
                hit = df_ncu.loc[((df_ncu['batch'] == batch) & (df_ncu['n_head'] == n_head) & (df_ncu['seq_len'] == seq_len) & (df_ncu['head_dim'] == head_dim) \
                                & (df_ncu['is_flash'] == True) & (df_ncu['q_seq_len'] == q_seq_len))]
                if len(hit) > 0:
                    df_nvml.loc[idx, 'kernel_name'] = hit['kernel_name'].values[0]
                    df_nvml.loc[idx, 'block_size'] = hit['block_size'].values[0]
                    df_nvml.loc[idx, 'grid_size'] = hit['grid_size'].values[0]
                    df_nvml.loc[idx, 'max_concurrent_block'] = hit['max_concurrent_block'].values[0]
                    df_nvml.loc[idx, 'found_ncu'] = True
                else:
                    print("No NCU entry found for this row:")
                    print(row)
                    df_nvml.loc[idx, 'found_ncu'] = False

            
    df_nvml = df_nvml.loc[df_nvml['is_flash'] == True]
    if ncu_transfer:
        df_nvml = df_nvml.loc[df_nvml['found_ncu'] == True]
    df_nvml.to_csv(save_to, index=False)

if __name__ == '__main__':
    main()
