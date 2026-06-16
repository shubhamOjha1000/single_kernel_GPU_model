import os
import csv
import copy
import argparse

import numpy as np
import pandas as pd
import sympy
import re

from utils import *

def energy_ncu_join(energy_csv, ncu_csv, gemm_mode, ncu_key_to_keep):
    df = convert_to_dataframe(energy_csv)
    if 'batch' in list(df.columns):
        df['batch'] = df['batch'].fillna(1)
    col_list = get_unique_configurations(df)
    time_dict, energy_dict, freq_dict = collect_time_energy(df, col_list)
    # print(time_dict)

    ncu_df = convert_to_dataframe(ncu_csv)
    ncu_df['energy'] = None
    ncu_df['time'] = None
    ncu_df['avg_freq'] = None
    keys = ncu_df.columns.tolist()
    for item in ncu_key_to_keep:
        keys.remove(item)
    ncu_df.drop(columns=keys, axis=1, inplace=True)

    for key, value in energy_dict.items():
        # print(value)
        keyparse = key.split('_')
        
        if (len(keyparse) == 8):
            batch = 1
            M = int(keyparse[0][4:])
            N = int(keyparse[1][4:])
            K = int(keyparse[2][4:])
            precM = keyparse[3][5:]
            precA = keyparse[4][5:]
            useTensorCore = keyparse[5][13:]
            initOption = int(keyparse[6][10:])
            numIter = int(keyparse[7][7:])
            trn = 'nn'
        elif (len(keyparse) == 9):
            batch = 1
            M = int(keyparse[0][4:])
            N = int(keyparse[1][4:])
            K = int(keyparse[2][4:])
            trn = keyparse[3][3:]
            precM = keyparse[4][5:]
            precA = keyparse[5][5:]
            useTensorCore = keyparse[6][13:]
            initOption = int(keyparse[7][10:])
            numIter = int(keyparse[8][7:])
        else:
            # print(keyparse)
            if (keyparse[0][5:] == 'nan'):
                batch = 1
            else:
                try:
                    batch = int(keyparse[0][5:])
                except:
                    batch = int(float(keyparse[0][5:]))
            M = int(keyparse[1][4:])
            N = int(keyparse[2][4:])
            K = int(keyparse[3][4:])
            trn = keyparse[4][3:]
            precM = keyparse[5][5:]
            precA = keyparse[6][5:]
            useTensorCore = keyparse[7][13:]
            initOption = int(keyparse[8][10:])
            numIter = int(keyparse[9][7:])

        ncu_df.loc[(ncu_df['gemm'].isin(gemm_mode)) & \
            (ncu_df['dimM'] == M) & \
            (ncu_df['dimN'] == N) & \
            (ncu_df['dimK'] == K) & \
            (ncu_df['trans'] == trn) & \
            (ncu_df['precM'] == precM) & \
            (ncu_df['precA'] == precA) & \
            (ncu_df['useTensorCore'] == eval(useTensorCore)) & \
            (ncu_df['batch'] == batch), 'energy'] = np.asarray(value).mean()
        
        ncu_df.loc[(ncu_df['gemm'].isin(gemm_mode)) & \
            (ncu_df['dimM'] == M) & \
            (ncu_df['dimN'] == N) & \
            (ncu_df['dimK'] == K) & \
            (ncu_df['trans'] == trn) & \
            (ncu_df['precM'] == precM) & \
            (ncu_df['precA'] == precA) & \
            (ncu_df['useTensorCore'] == eval(useTensorCore)) & \
            (ncu_df['batch'] == batch), 'time'] = np.asarray(time_dict[key]).mean()
        
        ncu_df.loc[(ncu_df['gemm'].isin(gemm_mode)) & \
            (ncu_df['dimM'] == M) & \
            (ncu_df['dimN'] == N) & \
            (ncu_df['dimK'] == K) & \
            (ncu_df['trans'] == trn) & \
            (ncu_df['precM'] == precM) & \
            (ncu_df['precA'] == precA) & \
            (ncu_df['useTensorCore'] == eval(useTensorCore)) & \
            (ncu_df['batch'] == batch), 'avg_freq'] = np.asarray(freq_dict[key]).mean()
    
    # ncu_df.to_csv(save_to, index=False)
    # if time/energy are empty, drop them
    ncu_df = ncu_df.loc[(ncu_df['time'] > 0) & (ncu_df['energy'] > 0)]
    return ncu_df

def get_symbols(expr, na_symbol, math_symbols, replace_dot):
    # if expr == na_symbol return empty
    if expr == na_symbol or expr == float('nan'):
        return []
    else:
        expr = expr.replace(' ', '') # remove any empty spaces
        re_string = '|'.join(map(re.escape, math_symbols))
        split = re.split(re_string, expr)
        split = list(filter(None, split))

        # Replace any '.' with replace_dot (otherwise . cannot be parsed by sympy)
        split = [x.replace('.', replace_dot) for x in split]
        _split = []
        for x in split:
            try:
                if (type(eval(x)) == int) or (type(eval(x)) == float):
                    # print(eval(x))
                    continue
                else:
                    _split.append(x)
            except:
                _split.append(x)
        return _split
    
def parse_cupti(df):
    df['ncu'] = df['ncu'].astype(str)
    na_symbol = 'nan'
    math_symbols = ['+', '-', '*', '/', '(', ')']
    replace_dot = '__'
    # df['ncu'].fillna(na_symbol)
    df['ncu_list'] = df['ncu'].apply(lambda x: get_symbols(x, na_symbol, math_symbols, replace_dot))

def get_function(df, cupti_name, replace_dot):
    entry = df.loc[df['nvprof'] == cupti_name]
    if len(entry) == 0:
        raise ValueError("Not supported cupti/nvprof field!")
    symbol_list = sympy.symbols(entry['ncu_list'].values[0])
    function_str = entry['ncu'].values[0].replace('.', replace_dot)
    func = sympy.sympify(function_str)
    return symbol_list, func

def generate_lut_from_nvml_ncu_metrics(energy_csv_paths, ncu_csv_paths, ncu_metrics_csv_paths, cupti_ncu_csv_path, save_to, cupti_parsing=True):

    gemm_mode = ['sgemm', 'sgemmbatched', 'gemmex']
    ncu_key_to_keep = ['gemm', 'batch', 'dimM', 'dimN', 'dimK', 'trans', 'precM', 'precA', 'useTensorCore', 'kernel_name', 'grid_size', 'block_size', \
                       'Elapsed Cycles', 'SM Active Cycles', 'Threads', 'Waves Per SM', 'Block Limit SM', 'Block Limit Registers', 'Block Limit Shared Mem', 'Block Limit Warps']

    ncu_df = energy_ncu_join(energy_csv_paths, ncu_csv_paths, gemm_mode, ncu_key_to_keep)
    ncu_df['max_concurrent_block'] = ncu_df.apply(lambda row: min(row['Block Limit SM'], row['Block Limit Registers'], \
                                                                  row['Block Limit Shared Mem'], row['Block Limit Warps']), axis=1)
    
    metrics_df = convert_to_dataframe(ncu_metrics_csv_paths)

    common_column_name = ['gemm', 'batch', 'dimM', 'dimN', 'dimK', 'trans', 'precM', 'precA', 'useTensorCore']

    if cupti_parsing:
        # Get CUPTI translation df
        df_cupti = pd.read_csv(cupti_ncu_csv_path)
        parse_cupti(df_cupti)

        cupti_metrics = df_cupti['nvprof'].tolist()
        replace_dot = '__'
        
        column_name = metrics_df.columns.tolist()
        column_to_rename = list(set(column_name) - set(common_column_name))
        rename_dict = {}
        for name in column_to_rename:
            rename_dict[name] = name.replace('.', replace_dot)
        metrics_df = metrics_df.rename(columns=rename_dict)
        merged = ncu_df.merge(metrics_df, on=common_column_name)

        # compute cupti metrics
        key_to_func = {}
        for key in cupti_metrics:
            symbol_list, func = get_function(df_cupti, key, replace_dot)
            key_to_func[key] = (symbol_list, func)
            merged[key] = None
            # get symbol list and function from key --> compute

        to_num = lambda x: eval(x.replace(',', '')) if type(x) == str else x
        for idx, row in merged.iterrows():
            for key in cupti_metrics:
                try:
                    symbol, f = key_to_func[key]
                    # print(symbol[0], str(symbol[0]))
                    sub_list = [(x, to_num(row[str(x)])) for x in symbol]
                    result = f.subs(sub_list)
                    merged.at[idx, key] = result
                except:
                    continue

        # remove original ncu metrics from df
        # columns_to_drop = rename_dict.values()
        # merged.drop(columns=columns_to_drop, axis=1, inplace=True)

    else:
        merged = ncu_df.merge(metrics_df, on=common_column_name)


    merged.to_csv(save_to, index=False)

    return

def generate_lut_from_nvml_ncu(energy_csv_paths, ncu_csv_paths, save_to, drop_cycles=False):
    gemm_mode = ['sgemm', 'sgemmbatched', 'gemmex']
    ncu_key_to_keep = ['gemm', 'batch', 'dimM', 'dimN', 'dimK', 'trans', 'precM', 'precA', 'useTensorCore', 'kernel_name', 'grid_size', 'block_size', \
                       'Elapsed Cycles', 'SM Active Cycles', 'Threads', 'Waves Per SM', 'Block Limit SM', 'Block Limit Registers', 'Block Limit Shared Mem', 'Block Limit Warps']
    if drop_cycles:
        ncu_key_to_keep.remove('Elapsed Cycles')
        ncu_key_to_keep.remove('SM Active Cycles')

    ncu_df = energy_ncu_join(energy_csv_paths, ncu_csv_paths, gemm_mode, ncu_key_to_keep)
    ncu_df['max_concurrent_block'] = ncu_df.apply(lambda row: min(row['Block Limit SM'], row['Block Limit Registers'], \
                                                                  row['Block Limit Shared Mem'], row['Block Limit Warps']), axis=1)
    ncu_df.to_csv(save_to, index=False)
    return

# LUT without NCU
def generate_lut_from_nvml_only(energy_csv_paths, save_to):
    # there can be multiple repeats -> average across different random trials
    df = convert_to_dataframe(energy_csv_paths)
    if 'batch' in list(df.columns):
        df['batch'] = df['batch'].fillna(1)
    
    # get unique configurations
    fields = ['batch', 'dimM', 'dimN', 'dimK', 'trn', 'precM', 'precA', 'useTensorCore', 'initOption', 'numIter']
    _fields = copy.deepcopy(fields)
    if 'trn' not in list(df.columns):
        _fields.remove('trn')

    if 'batch' not in list(df.columns):
        _fields.remove('batch')

    new_df = df[_fields].drop_duplicates()

    # construct a new df from this unique column dict
    # new_df = pd.DataFrame.from_dict(col_dict)

    # print(list(new_df.columns))
    new_df = new_df.rename(columns={'trn': 'trans'})
    new_df['energy'] = 0
    new_df['time'] = 0
    new_df['avg_freq'] = 0
    # print(list(new_df.columns))

    # print(_fields)
    # print(list(new_df.columns))
    for idx, row in new_df.iterrows():

        if ('trn' in _fields) and ('batch' in _fields):
            df_slice = df.loc[(df['batch'] == row['batch']) & \
                              (df['dimM'] == row['dimM']) & \
                              (df['dimN'] == row['dimN']) & \
                              (df['dimK'] == row['dimK']) & \
                              (df['trn'] == row['trans']) & \
                              (df['precM'] == row['precM']) & \
                              (df['precA'] == row['precA']) & \
                              (df['useTensorCore'] == row['useTensorCore']) & \
                              (df['initOption'] == row['initOption']) & \
                              (df['numIter'] == row['numIter'])]
            
            time = np.asarray(df_slice['time_per_iter'].values).mean()
            energy = np.asarray(df_slice['energy_per_iter'].values).mean()

            # avg_temp,avg_freq
            freq = np.asarray(df_slice['avg_freq'].values).mean()

            new_df.loc[(new_df['batch'] == row['batch']) & \
                       (new_df['dimM'] == row['dimM']) & \
                       (new_df['dimN'] == row['dimN']) & \
                       (new_df['dimK'] == row['dimK']) & \
                       (new_df['trans'] == row['trans']) & \
                       (new_df['precM'] == row['precM']) & \
                       (new_df['precA'] == row['precA']) & \
                       (new_df['useTensorCore'] == row['useTensorCore']) & \
                       (new_df['initOption'] == row['initOption']) & \
                       (new_df['numIter'] == row['numIter']), 'energy'] = energy
            new_df.loc[(new_df['batch'] == row['batch']) & \
                       (new_df['dimM'] == row['dimM']) & \
                       (new_df['dimN'] == row['dimN']) & \
                       (new_df['dimK'] == row['dimK']) & \
                       (new_df['trans'] == row['trans']) & \
                       (new_df['precM'] == row['precM']) & \
                       (new_df['precA'] == row['precA']) & \
                       (new_df['useTensorCore'] == row['useTensorCore']) & \
                       (new_df['initOption'] == row['initOption']) & \
                       (new_df['numIter'] == row['numIter']), 'time'] = time
            new_df.loc[(new_df['batch'] == row['batch']) & \
                       (new_df['dimM'] == row['dimM']) & \
                       (new_df['dimN'] == row['dimN']) & \
                       (new_df['dimK'] == row['dimK']) & \
                       (new_df['trans'] == row['trans']) & \
                       (new_df['precM'] == row['precM']) & \
                       (new_df['precA'] == row['precA']) & \
                       (new_df['useTensorCore'] == row['useTensorCore']) & \
                       (new_df['initOption'] == row['initOption']) & \
                       (new_df['numIter'] == row['numIter']), 'avg_freq'] = freq
        
        else:
            raise NotImplementedError('Benchmark configuration should have transpose and batch keys.')
        
    new_df.to_csv(save_to, index=False)
    return

def main():

    # argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_to', required=True, help='folder you want to save the summary result')
    parser.add_argument('--nvml_path', required=True, nargs='*', help='folder for nvml results')
    parser.add_argument('--enable_ncu', action='store_true', default=False, help='NCU results are available?')
    parser.add_argument('--ncu_path', nargs='*', help='if enable_ncu, specify the folder for ncu results')
    parser.add_argument('--ncu_metrics_path', nargs='*', help='if ncu metrics path is different from ncu_path')
    parser.add_argument('--ncu_metrics_yaml', nargs='*', help='if enable_ncu and ncu metrics are collectd, specify the yaml file that lists the collected ncu metrics')
    parser.add_argument('--ncu_cupti_parse', default=False, action='store_true', help='obtain CUPTI translated values from NCU?')
    parser.add_argument('--ncu_to_cupti_csv', help='if enable_ncu and ncu metrics are collected, specify the ncu to cupti metric conversion rule csv file')
    parser.add_argument('--ncu_drop_cycles', default=False, action='store_true', help='drop ncu elapsed/sm active cycles')

    args = parser.parse_args()

    # if the save_to folder doesn't exist, create one
    if not os.path.exists(args.save_to):
        os.mkdir(args.save_to)

    if args.ncu_metrics_path is None:
        args.ncu_metrics_path = args.ncu_path

    # call parsers
    try:
        for idx, p in enumerate(args.nvml_path):
            save_to_path = os.path.join(args.save_to, 'nvml_{}.csv'.format(idx))
            os.system('python3 parse_folder.py --save_to {} --path_to_folder {} --get_temp_freq'.format(save_to_path, p))
    except:
        print("Parsing NVML results with parse_folder.py failed. Check the python script and the arguments!")
        exit()

    if args.enable_ncu:
        try:
            for idx, p in enumerate(args.ncu_path):
                save_to_path = os.path.join(args.save_to, 'ncu_{}.csv'.format(idx))
                os.system('python3 ncu_parser.py --save_to {} --path_to_folder {}'.format(save_to_path, p))
        except:
            print("Parsing NCU results with ncu_parser.py failed. Check the python script and the arguments!")
            exit()

        if args.ncu_metrics_yaml is not None:
            try:
                for idx, p in enumerate(args.ncu_metrics_path):
                    save_to_path = os.path.join(args.save_to, 'ncu_metrics_{}.csv'.format(idx))
                    metrics_yaml_str = ' '.join(args.ncu_metrics_yaml)
                    os.system('python3 ncu_metrics_parser.py --save_to {} --path_to_folder {} --ncu_metrics {}'.format(save_to_path, p, metrics_yaml_str))
            except:
                print("Parsing NCU metrics results with ncu_parser.py failed. Check the python script and the arguments!")
                exit()

    # generate lut
    energy_csv_paths = [os.path.join(args.save_to, 'nvml_{}.csv'.format(i)) for i in range(len(args.nvml_path))]
    lut_path = os.path.join(args.save_to, 'lut.csv')

    if not args.enable_ncu:
        generate_lut_from_nvml_only(energy_csv_paths, lut_path)
    
    else:
        ncu_csv_paths = [os.path.join(args.save_to, 'ncu_{}.csv'.format(i)) for i in range(len(args.ncu_path))]
        
        if args.ncu_metrics_yaml is not None:
            ncu_metrics_csv_paths = [os.path.join(args.save_to, 'ncu_metrics_{}.csv'.format(i)) for i in range(len(args.ncu_metrics_path))]
            generate_lut_from_nvml_ncu_metrics(energy_csv_paths, ncu_csv_paths, ncu_metrics_csv_paths, args.ncu_to_cupti_csv, lut_path, cupti_parsing=args.ncu_cupti_parse)
        
        else:
            generate_lut_from_nvml_ncu(energy_csv_paths, ncu_csv_paths, lut_path, args.ncu_drop_cycles)
        
    
if __name__ == '__main__':
    main()




            
        


