import os
import sys
import argparse

import pandas as pd

import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload_csv_path', required=True)
    parser.add_argument('--bash_save_path', required=True)
    parser.add_argument('--nccl_save_path', required=True)
    parser.add_argument('--nccl_save_subpath', required=True)
    parser.add_argument('--n_gpus', default=2, type=int)
    parser.add_argument('--random_trials', default=1, type=int)
    parser.add_argument('--blocking', default=False, action='store_true')
    parser.add_argument('--lock_gpu_clock', default=False, action='store_true')
    parser.add_argument('--gpu_clock_freq', default=1410, type=int, help='gpu clock frequency to be locked in MHz')

    args = parser.parse_args()

    # Workload csv file should define:
    # - torch.distributed operation type
    # - source, destination (only applicable to certain operation types)
    # - torch.tensor size (1/2/3 dims)
    # - number of iterations for power reading
    workload = pd.read_csv(args.workload_csv_path)

    # Bash file
    # If a bash file with the same name exist, delete previous one
    if os.path.exists(args.bash_save_path):
        os.remove(args.bash_save_path)

    # Create a bash file
    with open(args.bash_save_path, 'w') as f:
        f.write('''\
#! /bin/bash
SECONDS=0
from_cli=("$@")
SUDO_PWD=${{from_cli[0]}}
NCCL_FOLDER="{}"
if [ ! -d "$NCCL_FOLDER" ]; then
mkdir $NCCL_FOLDER
fi\n
'''.format(args.nccl_save_path))
            
        if args.lock_gpu_clock:
            f.write('''\
echo ${{SUDO_PWD}} | sudo -S nvidia-smi -lgc {},{}\n
'''.format(args.gpu_clock_freq, args.gpu_clock_freq))
        
        for random in range(args.random_trials):

            f.write('''\
NCCL_SUBFOLDER="{}/{}_{}"
if [ ! -d "$NCCL_SUBFOLDER" ]; then
mkdir $NCCL_SUBFOLDER
fi\n'''.format(args.nccl_save_path, args.nccl_save_subpath, random))
            
            for _, config in workload.iterrows():
                op = config['op']
                root = config['root']
                size = config['size']
                iter = config['iter']
                in_place = config['in_place']

                result_file = '{}/{}_{}/op_{}_root_{}_size_{}_iter_{}_inplace_{}.csv'.format(args.nccl_save_path, args.nccl_save_subpath, random, \
                                                                                             op, root, size, iter, in_place)
                f.write('./build/{}_perf -g {} -b {} -e {} -f 2 -n {} -r {} -F {} -I {} -z {}\n'.format(op, args.n_gpus, size, size, iter, root, result_file, 1 if in_place else 0, 1 if args.blocking else 0))
                
        if args.lock_gpu_clock:
            f.write('''\
echo ${SUDO_PWD} | sudo -S nvidia-smi -rgc\n
''')
                   
        f.write('''
duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."
''')
    
if __name__ == '__main__':
    main()
