import os
import shutil
import argparse

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import re
import math

def get_equivalent_gemm_workload(b, m, c, hw, rs, stride, padding, prec):
    # cuDNN implicit gemm
    pq = math.floor((hw + 2 * padding - rs) / stride + 1)
    query = {}
    query['batch'] = 1
    query['dimM'] = b * pq * pq
    query['dimN'] = m
    query['dimK'] = c * rs * rs
    query['precM'] = prec
    query['precA'] = prec
    query['useTensorCore'] = True

    return query

def parse_name_and_get_stats(row, workload_name, folder, return_nvml=False):
    batch = re.findall('b[0-9]+', workload_name)
    m = re.findall('m[0-9]+', workload_name)
    c = re.findall('c[0-9]+', workload_name)
    hw = re.findall('hw[0-9]+', workload_name)
    rs = re.findall('rs[0-9]+', workload_name)
    stride = re.findall('stride[0-9]+', workload_name)
    padding = re.findall('padding[0-9]+', workload_name)

    if len(batch)==0:
        batch = 1
    else:
        batch = int(batch[0][1:])
    
    m = int(m[0][1:])
    c = int(c[0][1:])
    hw = int(hw[0][2:])
    rs = int(rs[0][2:])
    stride = int(stride[0][6:])
    padding = int(padding[0][7:])

    prec = 'fp32' if 'fp32' in workload_name else ('fp16' if 'fp16' in workload_name else 'bf16')

    # kernel name from ncu
    ncu_file_name = 'ncu_nn_conv_conv2d_{}_b{}_m{}_c{}_hw{}_rs{}_stride{}_padding{}.csv'.format(prec, batch, m, c, hw, rs, stride, padding)
    try:
        ncu_df = pd.read_csv(os.path.join(folder, 'ncu', ncu_file_name))
    except:
        return {}

    # Find the kernel corresponding to GEMM operatio in conv2d
    # name should have either: cutlass, gemm 
    ncu_df['is_gemm_kernel'] = ncu_df['kernel_name'].apply(lambda x: ('gemm' in x.lower()) or ('cutlass' in x.lower()) or ('cudnn_ampere' in x.lower()) or ('xmma' in x.lower())) 
    gemm = ncu_df.loc[ncu_df['is_gemm_kernel'] == True]

    if len(gemm) == 0:
        return {}
    elif len(gemm) > 1:
        return {}

    kernel_name = gemm['kernel_name'].values[0]

    if 'magma' in kernel_name:
        return {}
    
    elapsed_cycles = gemm['Elapsed Cycles'].values[0]
    if type(elapsed_cycles) == str:
        elapsed_cycles = elapsed_cycles.replace(',', '')
    elapsed_cycles = float(elapsed_cycles)
    max_concurrent_block = int(gemm['max_concurrent_block'].values[0])
    block_size = eval(gemm['block_size'].values[0])
    grid_size = eval(gemm['grid_size'].values[0])

    ncu_df['cycles'] = ncu_df['Elapsed Cycles'].apply(lambda x : float(x.replace(',', '')) if type(x) == str else float(x))
    portion_of_gemm_cycles = elapsed_cycles / ncu_df['cycles'].sum()

    if not return_nvml:
        return {'b': batch, 'm': m, 'c': c, 'hw': hw, 'rs': rs, 'stride': stride, 'padding': padding, 'prec': prec, 'avg_freq': 900.0, \
                'kernel_name': kernel_name, 'elapsed_cycles': elapsed_cycles, \
                'max_concurrent_block': max_concurrent_block, 'block_size': block_size, 'grid_size': grid_size, \
                'portion_of_gemm_cycles': portion_of_gemm_cycles}
    else:
        return {'b': batch, 'm': m, 'c': c, 'hw': hw, 'rs': rs, 'stride': stride, 'padding': padding, 'time': row['time'], 'energy': row['energy'], \
                'avg_freq': row['avg_freq'], 'prec': prec, \
                'kernel_name': kernel_name, 'elapsed_cycles': elapsed_cycles, \
                'max_concurrent_block': max_concurrent_block, 'block_size': block_size, 'grid_size': grid_size, \
                'portion_of_gemm_cycles': portion_of_gemm_cycles}

def get_time_energy_for_op(df=None, op_name=None, folder=None):
    summary = []
    for idx, row in df.iterrows():
        workload_name = row['workload']
        if op_name not in workload_name:
            continue

        parsed = parse_name_and_get_stats(row, workload_name, folder, True)
        if len(parsed) > 0:
            summary.append(parsed)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_folder', type=str, required=True)
    parser.add_argument('--save_to', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.result_folder, 'nvml/nvml_parsed.csv'))

    summary = get_time_energy_for_op(df, 'conv', args.result_folder)
    summary = pd.DataFrame(summary)

    for idx, row in summary.iterrows():
        query = get_equivalent_gemm_workload(row['b'], row['m'], row['c'], row['hw'], row['rs'], \
                                            row['stride'], row['padding'], row['prec'])
        for key, value in query.items():
            summary.loc[idx, key] = value

    summary.to_csv(args.save_to, index=False)
    
if __name__ == '__main__':
    main()
