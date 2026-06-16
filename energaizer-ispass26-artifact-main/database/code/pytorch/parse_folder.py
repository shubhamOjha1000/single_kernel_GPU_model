import os
import csv
import argparse

import warnings
warnings.simplefilter(action='ignore')

import pandas as pd

from parse_utils import get_energy, get_ncu

def main():
    # argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_folder', required=True, help='path to the folder that has all csv files to be parsed')
    parser.add_argument('--save_to', required=True, help='filename you want to save the summary result')
    parser.add_argument('--drop_short_recording', default=False, action='store_true')
    parser.add_argument('--drop_long_recording', default=False, action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.save_to):
        os.mkdir(args.save_to)
    
    nvml_save_to = os.path.join(args.save_to, 'nvml')
    ncu_save_to = os.path.join(args.save_to, 'ncu')
    metrics_save_to = os.path.join(args.save_to, 'metrics')

    if not os.path.exists(nvml_save_to):
        os.mkdir(nvml_save_to)
    if not os.path.exists(ncu_save_to):
        os.mkdir(ncu_save_to)
    if not os.path.exists(metrics_save_to):
        os.mkdir(metrics_save_to)

    nvml_summary = []

    for subdir, _, files in os.walk(args.path_to_folder):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith('.csv'):
                # print("Found a csv file %s", filepath)
                filename = str(file).replace(".csv", "")
                fields = filename.split("_")

                if fields[0] == 'ncu':
                    _ = get_ncu(filepath, save_to=os.path.join(ncu_save_to, str(file)))
                elif fields[0] == 'metrics':
                    _ = get_ncu(filepath, save_to=os.path.join(metrics_save_to, str(file)))
                else:
                    # NVML
                    iter_str = [x for x in fields if 'iter' in x][0]
                    iter = eval(iter_str[4:])
                    energy, time, freq = get_energy(pd.read_csv(filepath), n_gpus=1)
                    temp = {'workload': str(file), \
                            'energy': energy[0] / iter, \
                            'time': time / iter * 1000., \
                            'avg_freq': freq[0]}
                    if ((not args.drop_short_recording) or (time > 5)) and ((not args.drop_long_recording) or (time < 40)):
                        nvml_summary.append(temp)
    
    # save nvml summary
    if len(nvml_summary) > 0:
        df = pd.DataFrame(nvml_summary)
        df.to_csv(os.path.join(nvml_save_to, 'nvml_parsed.csv'), index=False)

if __name__ == '__main__':
    main()
                    