import os
import shutil
import argparse

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import re
import math

def parse_name_and_get_stats(row, workload_name, folder, op_name, return_nvml=False):

    workload_name = workload_name.replace(op_name, '').replace('misc', '')

    d = re.findall('d[0-9]+', workload_name)
    d = int(d[0][1:])

    prec = 'fp32' if 'fp32' in workload_name else ('fp16' if 'fp16' in workload_name else 'bf16')

    # kernel name from ncu
    ncu_file_name = 'ncu_misc_{}_{}_d{}.csv'.format(op_name, prec, d)
    ncu_df = pd.read_csv(os.path.join(folder, 'ncu', ncu_file_name))

    # too large workloads might have multiple kernels consecutively launched - ignore them
    ncu_df['initialize_kernel'] = ncu_df['kernel_name'].apply(lambda x: ('distribution_elementwise_grid_stride_kernel' in x) and ('curand' in x))
    ncu_df = ncu_df.loc[ncu_df['initialize_kernel'] == False]

    if len(ncu_df) == 0:
        return {}
    
    ncu_df['is_valid_kernel'] = ncu_df['kernel_name'].apply(lambda x: 'elementwise' in x)
    ncu_df = ncu_df.loc[ncu_df['is_valid_kernel'] == True]
    ncu_df.reset_index(drop=True, inplace=True)

    if len(ncu_df) > 1:
        # print(workload_name)
        return {}
    
    kernel_name = ncu_df.loc[len(ncu_df)-1, 'kernel_name']

    cycle_key = 'Elapsed Cycles' if 'Elapsed Cycles' in ncu_df.columns else 'sm__cycles_elapsed.max'

    elapsed_cycles = ncu_df.loc[len(ncu_df)-1, cycle_key]
    if type(elapsed_cycles) == str:
        elapsed_cycles = elapsed_cycles.replace(',', '')
    elapsed_cycles = float(elapsed_cycles)
    max_concurrent_block = int(ncu_df.loc[len(ncu_df)-1, 'max_concurrent_block'])
    block_size = eval(ncu_df.loc[len(ncu_df)-1, 'block_size'])
    grid_size = eval(ncu_df.loc[len(ncu_df)-1, 'grid_size'])

    ncu_df['cycles'] = ncu_df['Elapsed Cycles'].apply(lambda x: float(x.replace(',', '')) if type(x)==str else float(x))
    total_cycles = float(ncu_df['cycles'].sum())

    if not return_nvml:
        return {'dim': d, 'prec': prec, 'avg_freq': 900.0, 'op': op_name, \
                'kernel_name': kernel_name, 'elapsed_cycles': elapsed_cycles, 'total_cycles': total_cycles, \
                'max_concurrent_block': max_concurrent_block, 'block_size': block_size, 'grid_size': grid_size}
    else:
        return {'dim': d, 'time': row['time'], 'energy': row['energy'], \
                'avg_freq': row['avg_freq'], 'prec': prec, 'op': op_name,\
                'kernel_name': kernel_name, 'elapsed_cycles': elapsed_cycles, 'total_cycles': total_cycles, \
                'max_concurrent_block': max_concurrent_block, 'block_size': block_size, 'grid_size': grid_size}

def get_time_energy_for_op(df=None, op_name=None, folder=None):
    summary = []
    for idx, row in df.iterrows():
        workload_name = row['workload']
        if op_name not in workload_name:
            continue

        parsed = parse_name_and_get_stats(row, workload_name, folder, op_name, True)
        if len(parsed) > 0:
            summary.append(parsed)

    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_folder', type=str, required=True)
    parser.add_argument('--save_to', type=str, required=True)
    args = parser.parse_args()

    df_all = pd.read_csv(os.path.join(args.result_folder, 'nvml/nvml_parsed.csv'))

    ops = ['pointwise_mul', 'pointwise_add', 'scalar_mul', 'scalar_add', 'typecast_from_bf16_to_fp32', 'typecast_from_fp32_to_bf16']

    dfs = {}
    for op in ops:
        temp = get_time_energy_for_op(df_all, op, args.result_folder)
        dfs[op] = pd.DataFrame(temp)

    _df = dfs['typecast_from_bf16_to_fp32']
    _df = _df.loc[(_df['prec'] == 'bf16')]
    _df['op'] = 'typecast_to_fp32'
    dfs['typecast_from_bf16_to_fp32'] = _df

    _df = dfs['typecast_from_fp32_to_bf16']
    _df = _df.loc[(_df['prec'] == 'fp32')]
    _df['op'] = 'typecast_to_bf16'
    dfs['typecast_from_fp32_to_bf16'] = _df

    df = pd.concat(dfs.values())
    df.to_csv(args.save_to, index=False)

if __name__ == '__main__':
    main()
