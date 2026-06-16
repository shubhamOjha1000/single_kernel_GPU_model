import os
import csv
import argparse

import warnings
warnings.simplefilter(action='ignore')

import pandas as pd

from parse import parse_for_gpus

def main():
    # argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_folder', required=True, help='path to the folder that has all csv files to be parsed')
    parser.add_argument('--save_to', required=True, help='filename you want to save the summary result')
    parser.add_argument('--n_gpus', default=2, type=int)

    args = parser.parse_args()

    # !!! File should be organized as
    # [args.path_to_folder]
    # |-- op_{}_root_{}_size_{}_iter_{}_inplace_{}.csv
    # !!! operation list: [p2p_block, p2p_nonblock, broadcast, scatter, gather, all_gather, reduce, all_reduce]
    op_list = ['broadcast', 'scatter', 'gather', 'all_gather', 'reduce', 'all_reduce']
    
    summary = []
    for subdir, _, files in os.walk(args.path_to_folder):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith('.csv'):
                print("Found a csv file %s", filepath)
                filename = str(file).replace(".csv", "") # remove .csv from the filename

                temp = {}
                temp['op'] = None
                # first, search for operation name
                for op in op_list:
                    if (filename.find(op) > 0):
                        temp['op'] = op
                
                if temp['op'] is None:
                    print("[WARN] Cannot find torch.distributed operation in the file.")
                    continue

                # next, get source, destimation, shape, and iteration information
                split_idx = filename.find('root')
                filename = filename[split_idx:]
                fields = filename.split('_') 
                # [0] -> root , [1] -> root we want
                # [2] -> size , [3] -> size (Bytes) we want
                # [4] -> iter , [5] -> iteration count we want
                # [6] -> inplace , [7] -> inplace flag
                temp['root'] = int(fields[1])
                temp['size'] = int(fields[3])
                temp['iter'] = int(fields[5])
                temp['inplace'] = eval(fields[7])

                total_bits = temp['size'] * 8 # Byte to bits

                # read timestamps (in seconds), and power (W) for each gpus
                try:
                    df = pd.read_csv(filepath, sep=',')
                except:
                    print("[WARN] Cannot read this csv file")
                    continue

                energy, time = parse_for_gpus(df, args.n_gpus)

                for i in range(args.n_gpus):
                    # normalize energy for per iteration
                    temp['energy_{}'.format(i)] = float(energy[i]) / 10**9 / float(temp['iter'])
                    temp['energy_per_bit_{}'.format(i)] = float(energy[i]) / 10**9 / float(temp['iter']) / float(total_bits) 
                
                temp['time'] = float(time[0]) / 10 **9 / float(temp['iter'])
                
                # Algorithmic bandwidth and bus bandwidth
                # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
                temp['bandwidth'] = float(temp['size']) / temp['time'] # Bytes / s (size simply divided by time)
                
                bus_bw_factor = 1.
                if temp['op'] == 'all_reduce':
                    bus_bw_factor = 2. * (args.n_gpus - 1) / args.n_gpus
                elif temp['op'] == 'all_gather':
                    bus_bw_factor = 1. * (args.n_gpus - 1) / args.n_gpus

                temp['bus_bandwidth'] = bus_bw_factor * temp['bandwidth']

                summary.append(temp)

    with open(args.save_to, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

                    
if __name__ == '__main__':
    main()
