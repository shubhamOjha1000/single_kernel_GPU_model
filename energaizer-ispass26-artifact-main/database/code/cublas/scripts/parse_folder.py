import os
import csv
import argparse

import warnings
warnings.simplefilter(action='ignore')

import pandas as pd

from utils import getPower

def main():
    # argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_folder', required=True, help='path to the folder that has all csv files to be parsed')
    parser.add_argument('--drop_zero_util', action='store_true')
    parser.add_argument('--save_to', required=True, help='filename you want to save the summary result')
    parser.add_argument('--get_temp_freq', default=False, action='store_true')

    args = parser.parse_args()

    # !!! the files should be organized as
    # - path_to_folder
    # |- {trial}_{dim_M}_{dim_N}_{dim_K}_{prec_m}_{prec_a}_{use_tensorcore}_{init_option}_{num_iter}.csv
    print("[INFO] The csv files in the folder should be names as follows: ")
    print("[INFO] [trial]_[dimM]_[dimN]_[dimK]_[precM]_[precA]_[useTensorCore]_[initOption]_[numIter].csv")
    print("[INFO] OR [trial]_[dimM]_[dimN]_[dimK]_[transpose]_[precM]_[precA]_[useTensorCore]_[initOption]_[numIter].csv")
    print("[INFO] OR [trial]_[batch]_[dimM]_[dimN]_[dimK]_[transpose]_[precM]_[precA]_[useTensorCore]_[initOption]_[numIter].csv")
    print("[INFO] If there exists any file that doesn't have this name format")
    print("[INFO] this script will skip that file.")

    # create a list of python dicts
    summary = []
    """
    fieldnames = ['trial', 'dimM', 'dimN', 'dimK', \
                  'precM' ,'precA', 'useTensorCore', 'initOption', 'numIter', \
                  'time', 'time_per_iter', 'energy', 'energy_per_iter', 'peak_power', 'avg_power', 'peak_util', 'drop_zero_util']
    # """
    
    # iterate through all files in this folder
    for subdir, _, files in os.walk(args.path_to_folder):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith('.csv'):
                print("Found a csv file %s", filepath)
                filename = str(file).replace(".csv", "") # remove .csv from the filename
                fields = filename.split('_')
                # print(fields)
                if ((len(fields) != 9) and (len(fields) != 10)) and (len(fields) != 11):
                    print("[WARN] This file name doesn't meet the requirement. Skipping this file.")
                    continue
                else:
                    if len(fields) == 9:
                        temp = {'trial': fields[0], \
                                'dimM': fields[1], \
                                'dimN': fields[2], \
                                'dimK': fields[3], \
                                'precM': fields[4], \
                                'precA': fields[5], \
                                'useTensorCore': fields[6], \
                                'initOption': fields[7], \
                                'numIter': fields[8]}
                    elif len(fields) == 10:
                        temp = {'trial': fields[0], \
                                'dimM': fields[1], \
                                'dimN': fields[2], \
                                'dimK': fields[3], \
                                'trn': fields[4], \
                                'precM': fields[5], \
                                'precA': fields[6], \
                                'useTensorCore': fields[7], \
                                'initOption': fields[8], \
                                'numIter': fields[9]}
                    elif len(fields) == 11:
                        temp = {'trial': fields[0], \
                                'batch': fields[1], \
                                'dimM': fields[2], \
                                'dimN': fields[3], \
                                'dimK': fields[4], \
                                'trn': fields[5], \
                                'precM': fields[6], \
                                'precA': fields[7], \
                                'useTensorCore': fields[8], \
                                'initOption': fields[9], \
                                'numIter': fields[10]}
                    
                    # read csv
                    try:
                        df = pd.read_csv(filepath, sep=', ')
                    except:
                        continue

                    # get power, total time
                    if args.get_temp_freq:
                        energy, time, peak_gpu_util, peak_power, temperature, freq = getPower(df, args.drop_zero_util, True)
                    else:
                        energy, time, peak_gpu_util, peak_power = getPower(df, args.drop_zero_util)

                    num_iter = float(fields[8]) if (len(fields) == 9) else (float(fields[9]) if (len(fields) == 10) else float(fields[10]))

                    temp['time'] = float(time) / 10**6 # ms
                    temp['time_per_iter'] = (float(time) / num_iter) / 10**6
                    temp['energy'] = float(energy) / 10**9 # J
                    temp['energy_per_iter'] = (float(energy) / num_iter) / 10**9 
                    temp['peak_power'] = peak_power # W
                    temp['avg_power'] = float(energy) / float(time) # W
                    temp['peak_util'] = peak_gpu_util # %
                    temp['drop_zero_util'] = args.drop_zero_util # boolean

                    if args.get_temp_freq:
                        temp['avg_temp'] = temperature
                        temp['avg_freq'] = freq

                    summary.append(temp)

    # Dump summary to a csv file
    with open(args.save_to, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

                    
if __name__ == '__main__':
    main()
