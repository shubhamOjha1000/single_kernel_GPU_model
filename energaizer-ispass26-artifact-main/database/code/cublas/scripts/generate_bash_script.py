import os
import sys
import argparse

import pandas as pd

import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload_csv_path', required=True)
    parser.add_argument('--bash_save_path', required=True)
    parser.add_argument('--profile_ncu', default=False, action='store_true')
    parser.add_argument('--profile_nvml', default=False, action='store_true')
    parser.add_argument('--skip_regular_ncu', default=False, action='store_true')
    parser.add_argument('--precisionM', default='fp32', type=str)
    parser.add_argument('--precisionA', default='fp32', type=str)
    parser.add_argument('--use_tensorcore', default=False, action='store_true')
    parser.add_argument('--use_sgemm', default=False, action='store_true')
    parser.add_argument('--nvml_save_path', type=str)
    parser.add_argument('--ncu_save_path', type=str)
    parser.add_argument('--sparsityA', default=0.0, type=float)
    parser.add_argument('--sparsityB', default=0.0, type=float)
    parser.add_argument('--all_transpose_options', default=False, action='store_true')
    parser.add_argument('--transpose_options', type=str, nargs='*')
    parser.add_argument('--cuda_device', default=0, type=int, help='Single GPU only')
    parser.add_argument('--num_trials', default=1, type=int)
    parser.add_argument('--ncu_bin_path', default='/usr/local/cuda/bin/ncu', type=str)
    parser.add_argument('--ncu_metrics', nargs='*', default=None)
    parser.add_argument('--lock_gpu_clock', default=False, action='store_true')
    parser.add_argument('--gpu_clock_freq', default=1410, type=int, help='gpu clock frequency to be locked in MHz')
    parser.add_argument('--set_power_limit', default=False, action='store_true')
    parser.add_argument('--gpu_power_limit', default=250, type=int)
    args = parser.parse_args()

    # Read csv file with configurations for workload to benchmark
    workload = pd.read_csv(args.workload_csv_path)
    columns = list(workload.columns)

    # Check if NCU metrics (.yaml) file is provided - if so, we will query those metrics in addition to the basic ncu
    if args.ncu_metrics is not None:
        metrics = []
        for p in args.ncu_metrics:
            with open(p) as f:
                _metrics = yaml.safe_load(f)['metrics']
                metrics.extend(_metrics)

    # If a bash file with the same name exist, delete previous one
    if os.path.exists(args.bash_save_path):
        os.remove(args.bash_save_path)

    # Create a bash file
    with open(args.bash_save_path, 'w') as f:
        f.write('''\
#! /bin/bash
SECONDS=0
from_cli=("$@")
SUDO_PWD=${from_cli[0]}\n
''')
        
        if args.profile_nvml:
            f.write('''\
NVML_FOLDER="{}"
if [ ! -d "$NVML_FOLDER" ]; then
mkdir $NVML_FOLDER
fi\n
'''.format(args.nvml_save_path))
        
        if args.profile_ncu:
            f.write('''\
NCU_FOLDER="{}"
if [ ! -d "$NCU_FOLDER" ]; then
mkdir $NCU_FOLDER
fi\n
'''.format(args.ncu_save_path))
            
        if args.lock_gpu_clock:
            f.write('''\
echo ${{SUDO_PWD}} | sudo -S nvidia-smi -i {} -lgc {},{}\n
'''.format(args.cuda_device, args.gpu_clock_freq, args.gpu_clock_freq))
            
        if args.set_power_limit:
            f.write('''\
echo ${{SUDO_PWD}} | sudo -S nvidia-smi -i {} -pl {}\n
'''.format(args.cuda_device, args.gpu_power_limit))
        
        # random trial counts
        sparsity_encountered = []
        for trial in range(args.num_trials):

            # information needed to execute nvml/ncu benchmarks
            # - batch, dim_M, dim_N, dim_K, iterations: should be defined in the csv workload file 
            # - sgemm option: if args.use_sgemm true, flag 'G' should be set
            # - precisionM, precisionA: if defined in csv workload file, use it. otherwise, use the default value in args.
            # - trn_matA, trn_matB: if defined in csv workload file, use it. 
            #                       if args.all_transpose_options is True, then all four combintaions are run
            #                       if args.all_transpose_options is False, then only run nn
            # - sparsityA, sparsityB: if defined in csv workload file, use it. otherwise, use the default value in args
            for _, config in workload.iterrows():
                if 'Batch' not in columns:
                    batch = 1
                else:
                    batch = config['Batch']

                dim_M = config['M']
                dim_N = config['N']
                dim_K = config['K']
                iter = config['Iterations']

                if 'PrecisionM' not in columns:
                    precM = args.precisionM
                else:
                    precM = config['PrecisionM']
                
                if 'PrecisionA' not in columns:
                    precA = args.precisionA
                else:
                    precA = config['PrecisionA']

                if 'Transpose' not in columns:
                    if args.all_transpose_options:
                        transpose = ['nn', 'nt', 'tn', 'tt']
                    elif len(args.transpose_options) > 0:
                        transpose = args.transpose_options
                    else:
                        transpose = ['nn']
                else:
                    transpose = [config['Transpose']]

                if 'Tensorcore' not in columns:
                    tensorcore = args.use_tensorcore
                else:
                    tensorcore = config['Tensorcore']
                
                if 'SparsityA' not in columns:
                    sparsityA = args.sparsityA
                else:
                    sparsityA = config['SparsityA']
                
                if 'SparsityB' not in columns:
                    sparsityB = args.sparsityB
                else:
                    sparsityB = config['SparsityB']

                # check if a folder for this sparsity configuraiton exists
                nvml_sparsity_subpath = "{}/spsA{}_spsB{}".format(args.nvml_save_path, \
                                                                  str(sparsityA).replace('.', 'p'), \
                                                                  str(sparsityB).replace('.', 'p'))
                ncu_sparsity_subpath = "{}/spsA{}_spsB{}".format(args.ncu_save_path, \
                                                                 str(sparsityA).replace('.', 'p'), \
                                                                 str(sparsityB).replace('.', 'p'))
                
                if (sparsityA, sparsityB) not in sparsity_encountered:
                    if args.profile_nvml:
                        f.write('''
NVML_SPARSITY_FOLDER="{}"
if [ ! -d "$NVML_SPARSITY_FOLDER" ]; then
mkdir $NVML_SPARSITY_FOLDER
fi\n
'''.format(nvml_sparsity_subpath))
                
                    if args.profile_ncu:
                        f.write('''
NCU_SPARSITY_FOLDER="{}"
if [ ! -d "$NCU_SPARSITY_FOLDER" ]; then
mkdir $NCU_SPARSITY_FOLDER
fi\n
'''.format(ncu_sparsity_subpath))
                        
                    sparsity_encountered.append((sparsityA, sparsityB))

                # iterate over all transpose options
                for trn in transpose:

                    exec_str = './bin/gemm_bench '
                    if args.use_sgemm:
                        exec_str += '-C '
                    elif (not tensorcore):
                        exec_str += '-C '
                    exec_str += '-M {} -A {} '.format(precM, precA)
                    # exec_str += '--sparsityA {} --sparsityB {} '.format(sparsityA, sparsityB)
                    exec_str += '--device {} '.format(args.cuda_device)
                        
                    if trn == 'nn':
                        exec_str = exec_str
                    elif trn == 'nt':
                        exec_str += '--trn_matB '
                    elif trn == 'tn':
                        exec_str += '--trn_matA '
                    else:
                        exec_str += '--trn_matA --trn_matB '
                    if batch > 1:
                        exec_str += '-B {} --strided_batched '.format(batch)
                    
                    nvml_result_file = '{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(
                        nvml_sparsity_subpath, trial, batch, dim_M, dim_N, dim_K, trn, \
                        precM, precA, 'False' if (args.use_sgemm or not tensorcore) else 'True', 0, iter
                    )
                    
                    nvml_exec_str = exec_str + '-I {} -O {} {} {} {}'.format(iter, \
                                                                             nvml_result_file, \
                                                                             dim_M, dim_N, dim_K)
                    
                    ncu_exec_str = exec_str + '-I 1 {} {} {}'.format(dim_M, dim_N, dim_K)

                    ncu_result_file = '{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(
                        ncu_sparsity_subpath, 'sgemm' if args.use_sgemm else 'gemmex', \
                        batch, dim_M, dim_N, dim_K, trn, precM, precA, \
                        'False' if (args.use_sgemm or not tensorcore) else 'True', 0
                    )

                    if args.profile_nvml:
                        f.write('CUDA_VISIBLE_DEVICES={} {}\n'.format(args.cuda_device, nvml_exec_str))
                    
                    # NCU doesn't have to be run multiple times
                    if args.profile_ncu and trial==0:
                        if not args.skip_regular_ncu:
                            f.write('echo ${{SUDO_PWD}} | sudo -S CUDA_VISIBLE_DEVICES={} {} --log-file {} --csv --set full --clock-control none {}\n'.format(
                                args.cuda_device, args.ncu_bin_path, ncu_result_file, ncu_exec_str
                            ))
                        if args.ncu_metrics is not None:
                            ncu_metrics_result_file = '{}/metrics_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(
                                ncu_sparsity_subpath, 'sgemm' if args.use_sgemm else 'gemmex', \
                                batch, dim_M, dim_N, dim_K, trn, precM, precA, \
                                'False' if (args.use_sgemm or not tensorcore) else 'True', 0
                            )
                            metrics_list = ''
                            for idx, entry in enumerate(metrics):
                                metrics_list += entry
                                if idx < len(metrics) - 1:
                                    metrics_list += ','
                            f.write('echo ${{SUDO_PWD}} | sudo -S CUDA_VISIBLE_DEVICES={} {} --log-file {} --csv --clock-control none --metrics {} {}\n'.format(
                                args.cuda_device, args.ncu_bin_path, ncu_metrics_result_file, metrics_list, ncu_exec_str
                            ))
                    
        f.write('''
duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."
''')
        
        if args.lock_gpu_clock:
            f.write('''\
echo ${{SUDO_PWD}} | sudo -S nvidia-smi -i {} -rgc\n
'''.format(args.cuda_device))
            
        if args.set_power_limit:
            f.write('''\
echo ${{SUDO_PWD}} | sudo -S nvidia-smi -i {} -pl 250\n
'''.format(args.cuda_device))

if __name__ == '__main__':
    main()
                    


