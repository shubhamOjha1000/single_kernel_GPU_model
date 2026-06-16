import os
import csv
import argparse

import warnings
warnings.simplefilter(action='ignore')

import yaml
import pandas as pd

def get_ncu(filename, save_to=None):
    # Count the number of rows that need to be removed
    skiprows=0
    with open(filename, 'r') as f:
        for line in f:
            vals = line.split(',"')
            if len(vals) == 1:
                skiprows += 1
            else:
                break
    
    df = pd.read_csv(filename, skiprows=skiprows)

    unique_ids = df['ID'].unique().tolist()

    summary = []
    for kernel_id in unique_ids:
        temp = {'kernel_id': kernel_id}
        df_subset = df.loc[df['ID'] == kernel_id]
        temp['kernel_name'] = df_subset['Kernel Name'].iloc[0]
        temp['block_size'] = df_subset['Block Size'].iloc[0]
        temp['grid_size'] = df_subset['Grid Size'].iloc[0]

        t_dict = pd.Series(df_subset['Metric Value'].values, index=df_subset['Metric Name']).to_dict()
        temp.update(t_dict)
        try:
            temp['max_concurrent_block'] = min(int(temp['Block Limit SM']), int(temp['Block Limit Registers']), \
                                               int(temp['Block Limit Shared Mem']), int(temp['Block Limit Warps']))
        except:
            pass
                
        summary.append(temp)
    
    if save_to is not None:
        to_write = pd.DataFrame(summary)
        to_write.to_csv(save_to, index=False)

    return summary

def get_energy(df, n_gpus):
    energy = []

    df['time_diff'] = df['timestamp'].diff()
    for i in range(n_gpus):
        df['energy_delta_{}'.format(i)] = df['time_diff'] * df['power_{}'.format(i)]
    
    for i in range(n_gpus):
        energy.append(df['energy_delta_{}'.format(i)].sum(axis=0, skipna=True))

    time = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    
    freq = []
    for i in range(n_gpus):
        freq.append(df['sm_clock_{}'.format(i)].mean(axis=0, skipna=True) * 1000.)
    
    return energy, time, freq