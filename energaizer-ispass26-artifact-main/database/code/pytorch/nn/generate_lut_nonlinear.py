import os
import shutil
import argparse

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import re
import math

def parse_name_and_get_stats(row, workload_name, folder, op_name, return_nvml=False, ncu_df_from_lut=None):
    batch = re.findall('b[0-9]+', workload_name)
    d = re.findall('d[0-9]+', workload_name)

    if len(batch)==0:
        batch = 1
    else:
        batch = int(batch[0][1:])
    
    d = int(d[0][1:])

    prec = 'fp32' if 'fp32' in workload_name else ('fp16' if 'fp16' in workload_name else 'bf16')

    # kernel name from ncu
    if ncu_df_from_lut is None:
        ncu_file_name = 'ncu_nn_nonlinear_{}_{}_b{}_d{}.csv'.format(op_name, prec, batch, d)
        ncu_df = pd.read_csv(os.path.join(folder, 'ncu', ncu_file_name))
        kernel_name = ncu_df.loc[len(ncu_df)-1, 'kernel_name']
        elapsed_cycles = ncu_df.loc[len(ncu_df)-1, 'Elapsed Cycles']
        if type(elapsed_cycles) == str:
            elapsed_cycles = elapsed_cycles.replace(',', '')
        elapsed_cycles = float(elapsed_cycles)
        max_concurrent_block = int(ncu_df.loc[len(ncu_df)-1, 'max_concurrent_block'])
        block_size = eval(ncu_df.loc[len(ncu_df)-1, 'block_size'])
        grid_size = eval(ncu_df.loc[len(ncu_df)-1, 'grid_size'])

        ncu_df['cycles'] = ncu_df['Elapsed Cycles'].apply(lambda x: float(x.replace(',', '')) if type(x) == str else float(x))
        total_cycles = float(ncu_df['cycles'].sum())
    else:
        ncu_row = ncu_df_from_lut.loc[(ncu_df_from_lut['batch']==batch) & (ncu_df_from_lut['dim']==d) & (ncu_df_from_lut['prec']==prec)]
        kernel_name = ncu_row['kernel_name'].values[0]
        elapsed_cycles = float(ncu_row['elapsed_cycles'].values[0])
        max_concurrent_block = int(ncu_row['max_concurrent_block'].values[0])
        block_size = eval(ncu_row['block_size'].values[0])
        grid_size = eval(ncu_row['grid_size'].values[0])
        total_cycles = float(ncu_row['total_cycles'].values[0])

    if not return_nvml:
        return {'batch': batch, 'dim': d, 'prec': prec, 'avg_freq': 900.0, \
                'kernel_name': kernel_name, 'elapsed_cycles': elapsed_cycles, 'total_cycles': total_cycles, \
                'max_concurrent_block': max_concurrent_block, 'block_size': block_size, 'grid_size': grid_size}
    else:
        return {'batch': batch, 'dim': d, 'time': row['time'], 'energy': row['energy'], \
                'avg_freq': row['avg_freq'], 'prec': prec, \
                'kernel_name': kernel_name, 'elapsed_cycles': elapsed_cycles, 'total_cycles': total_cycles, \
                'max_concurrent_block': max_concurrent_block, 'block_size': block_size, 'grid_size': grid_size}

def get_time_energy_for_op(df=None, op_name=None, folder=None, ncu_df_from_lut=None):

    summary = []

    if df is not None:
        for idx, row in df.iterrows():
            workload_name = row['workload']
            if op_name not in workload_name:
                continue

            summary.append(parse_name_and_get_stats(row, workload_name, folder, op_name, True, ncu_df_from_lut=ncu_df_from_lut))

    else:
        for subdir, _, files in os.walk(os.path.join(folder, 'ncu')):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith('.csv'):
                    workload_name = file
                    if op_name not in workload_name:
                        continue

                    summary.append(parse_name_and_get_stats(None, workload_name, folder, op_name, False))

    
    return summary

def parse_and_save(folder, operation_name, save_to='./', save_name_prefix='yz8', save_name_suffix='freq900_lut', ncu_df_from_lut=None):
    df = pd.read_csv(os.path.join(folder, 'nvml/nvml_parsed.csv'))
    summary = get_time_energy_for_op(df, operation_name, folder, ncu_df_from_lut=ncu_df_from_lut)
    summary = pd.DataFrame(summary)

    unique_prec = summary['prec'].unique()
    for prec in unique_prec:
        _df = summary.loc[summary['prec'] == prec]
        filename = save_name_prefix + "_{}_{}_".format(operation_name, prec) + save_name_suffix + '.csv'
        if os.path.isfile(os.path.join(save_to, filename)):
            original_df = pd.read_csv(os.path.join(save_to, filename))
            _df = pd.concat([_df, original_df]).drop_duplicates(subset=['batch', 'dim', 'prec'])

        _df.to_csv(os.path.join(save_to, filename), index=False)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_folder', type=str, required=True)
    parser.add_argument('--operation_name', choices=['layernorm', 'softmax'], required=True, type=str)
    parser.add_argument('--save_to', type=str, required=True)
    parser.add_argument('--save_prefix', type=str, default='a100', required=False)
    parser.add_argument('--save_suffix', type=str, default='lut', required=False)
    
    parser.add_argument('--transfer_ncu', default=False, action='store_true')
    parser.add_argument('--transfer_ncu_files', nargs='*', default=None)

    args = parser.parse_args()

    if args.transfer_ncu:
        files = []
        for p in args.transfer_ncu_files:
            files.append(pd.read_csv(p))
        ncu_df = pd.concat(files)
    else:
        ncu_df = None

    parse_and_save(args.result_folder, args.operation_name, \
                   args.save_to, args.save_prefix, args.save_suffix, ncu_df)
    
if __name__ == '__main__':
    main()
