import os
import csv
import argparse
import copy

import warnings
warnings.simplefilter(action='ignore')

import numpy as np
import pandas as pd

# Configuration parameters
fields = ['batch', 'dimM', 'dimN', 'dimK', 'trn', 'precM', 'precA', 'useTensorCore', 'initOption', 'numIter']

def getPower(df, nonzeroUtil=False, get_freq_temp=False):

    df['time_diff'] = df['timestamp'].diff()
    df['energy_delta'] = df['time_diff'] * df['power_draw_w']

    # Drop parts where GPU utilization is 0 (usually at the beginning and the end)
    if nonzeroUtil and not (df['utilization_gpu'].min(skipna=True) > 0):
        # Get the difference in the gpu utlization in each consecutive timestep,
        # then shift up with 1 row
        df['gpu_util_diff'] = df['utilization_gpu'].diff().shift(-1)

        # identify the first zero -> nonzero 
        if len(df[(df['gpu_util_diff'] > 0) & (df['utilization_gpu'] <= 0)]) > 0:
            # print(len(df[(df['gpu_util_diff'] > 0) & (df['utilization_gpu'] <= 0)]))
            zero_to_nonzero = df[(df['gpu_util_diff'] > 0) & (df['utilization_gpu'] <= 0)]['timestamp'].iloc[0]
        else:
            zero_to_nonzero = df['timestamp'].iloc[0]

        # shift back by one row
        df['gpu_util_diff'] = df['gpu_util_diff'].shift(1)

        # identify the last nonzero -> zero
        if len(df[(df['gpu_util_diff'] < 0) & (df['utilization_gpu'] <=0)]) > 0:
            nonzero_to_zero = df[(df['gpu_util_diff'] < 0) & (df['utilization_gpu'] <=0)]['timestamp'].iloc[-1]
        else:
            nonzero_to_zero = df['timestamp'].iloc[-1]

        df_nonzero = df[(df['timestamp'] > zero_to_nonzero) & (df['timestamp'] < nonzero_to_zero)]
        
        starttime = df['timestamp'].iloc[0]
        endtime = df['timestamp'].iloc[-1]
        print("Dropping timestamps until {} and from {} - zero GPU utlization".format(zero_to_nonzero, nonzero_to_zero))
        print("-- dropping total {:.2f}ms".format(float(zero_to_nonzero - starttime + endtime - nonzero_to_zero) / 10**6))

        df = df_nonzero
    
    energy = df['energy_delta'].sum(axis=0, skipna=True)
    time = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    peak_power = df['power_draw_w'].max(skipna=True)
    peak_gpu_util = df['utilization_gpu'].max(skipna=True)

    if get_freq_temp:
        try:
            avg_temp = df['temperature_gpu'].mean(axis=0, skipna=True)
            avg_freq = df['clocks_current_sm_mhz'].mean(axis=0, skipna=True)
        except:
            avg_temp = -1
            avg_freq = -1
        return energy, time, peak_gpu_util, peak_power, avg_temp, avg_freq

    return energy, time, peak_gpu_util, peak_power

def convert_to_dataframe(filenames=[]):
    """
    Get all csv files listed in filenames, 
    and return one pandas dataframe 
    """
    df = None
    for fname in filenames:
        df_temp = pd.read_csv(fname)
        if df is None:
            df = df_temp
        else:
            df = pd.concat([df, df_temp], ignore_index=True)
    return df

def get_unique_configurations(df):
    """
    Return a list of unique configurations in df
    """
    _fields = copy.deepcopy(fields)
    if 'trn' not in list(df.columns):
        _fields.remove('trn')

    if 'batch' not in list(df.columns):
        _fields.remove('batch')

    _df = df[_fields].drop_duplicates()
    col_dict = _df.to_dict('index')
    col_list = [col_dict[k] for k in col_dict.keys()]
    return col_list

def collect_time_energy(df, col_list):
    """
    Return energy and time measured for all unique configurations.
    When there are multiple trial runs, all values will be returned for both energy/time.
    """
    time_dict = {}
    energy_dict = {}
    freq_dict = {}
    for config in col_list:
        if ('trn' in list(df.columns)) and ('batch' in list(df.columns)):
            
            df_slice = df.loc[(df['dimM'] == config['dimM']) & \
                            (df['dimN'] == config['dimN']) & \
                            (df['dimK'] == config['dimK']) & \
                            (df['trn'] == config['trn']) & \
                            (df['precM'] == config['precM']) & \
                            (df['precA'] == config['precA']) & \
                            (df['useTensorCore'] == config['useTensorCore']) &\
                            (df['initOption'] == config['initOption']) & \
                            (df['numIter'] == config['numIter']) & \
                            (df['batch'] == config['batch'])]
            k = "batch{}_dimM{}_dimN{}_dimK{}_trn{}_precM{}_precA{}_useTensorCore{}_initOption{}_numIter{}".format(config['batch'], \
                                                                                                                    config['dimM'], \
                                                                                                                    config['dimN'], \
                                                                                                                    config['dimK'], \
                                                                                                                    config['trn'], \
                                                                                                                    config['precM'], \
                                                                                                                    config['precA'], \
                                                                                                                    config['useTensorCore'], \
                                                                                                                    config['initOption'], \
                                                                                                                    config['numIter'])
            # print(k)
            time_dict[k] = df_slice['time_per_iter'].values
            energy_dict[k] = df_slice['energy_per_iter'].values
            freq_dict[k] = df_slice['avg_freq'].values
        elif ('trn' in list(df.columns)):
            df_slice = df.loc[(df['dimM'] == config['dimM']) & \
                            (df['dimN'] == config['dimN']) & \
                            (df['dimK'] == config['dimK']) & \
                            (df['trn'] == config['trn']) & \
                            (df['precM'] == config['precM']) & \
                            (df['precA'] == config['precA']) & \
                            (df['useTensorCore'] == config['useTensorCore']) &\
                            (df['initOption'] == config['initOption']) & \
                            (df['numIter'] == config['numIter'])]
            k = "dimM{}_dimN{}_dimK{}_trn{}_precM{}_precA{}_useTensorCore{}_initOption{}_numIter{}".format(config['dimM'], \
                                                                                                           config['dimN'], \
                                                                                                           config['dimK'], \
                                                                                                           config['trn'], \
                                                                                                           config['precM'], \
                                                                                                           config['precA'], \
                                                                                                           config['useTensorCore'], \
                                                                                                           config['initOption'], \
                                                                                                           config['numIter'])
            time_dict[k] = df_slice['time_per_iter'].values
            energy_dict[k] = df_slice['energy_per_iter'].values
            freq_dict[k] = df_slice['avg_freq'].values
        else:
            df_slice = df.loc[(df['dimM'] == config['dimM']) & \
                            (df['dimN'] == config['dimN']) & \
                            (df['dimK'] == config['dimK']) & \
                            (df['precM'] == config['precM']) & \
                            (df['precA'] == config['precA']) & \
                            (df['useTensorCore'] == config['useTensorCore']) &\
                            (df['initOption'] == config['initOption']) & \
                            (df['numIter'] == config['numIter'])]
            k = "dimM{}_dimN{}_dimK{}_precM{}_precA{}_useTensorCore{}_initOption{}_numIter{}".format(config['dimM'], \
                                                                                                    config['dimN'], \
                                                                                                    config['dimK'], \
                                                                                                    config['precM'], \
                                                                                                    config['precA'], \
                                                                                                    config['useTensorCore'], \
                                                                                                    config['initOption'], \
                                                                                                    config['numIter'])
            time_dict[k] = df_slice['time_per_iter'].values
            energy_dict[k] = df_slice['energy_per_iter'].values
            freq_dict[k] = df_slice['avg_freq'].values

    return time_dict, energy_dict, freq_dict

def get_all_config_list(fields=[], col_list=None):
    """
    In col_list, get a subset of unique configurations specified in fields.
    """
    if len(fields) == 0:
        raise ValueError("There is no field provided!")
    
    if col_list is None:
        raise ValueError("Empty column list provided!")
    
    # Get all combinations of keys listed in fields
    def get_sub_dict(x={}, keys=[]):
        return dict((k, x[k]) for k in keys)
    
    combinations = [get_sub_dict(x, fields) for x in col_list]

    # return list({x[fields[0]]:x for x in combinations}.values())
    return [dict(y) for y in set(tuple(x.items()) for x in combinations)]
