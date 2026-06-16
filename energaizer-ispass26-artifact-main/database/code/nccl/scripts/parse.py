import os
import csv
import argparse

import warnings
warnings.simplefilter(action='ignore')

import numpy as np
import pandas as pd

def parse_for_gpus(df, ngpus=2):
    timestamp_key = 'timestamp'
    power_key = 'power_draw_w'

    energy = []
    time = []

    for i in range(ngpus):
        _timestamp = timestamp_key + "_" + str(i)
        _power = power_key + "_" + str(i)
        _timediff = 'timediff' + "_" + str(i)
        _energy_delta = 'energy_delta' + "_" + str(i)

        df[_timediff] = df[_timestamp].diff()
        df[_energy_delta] = df[_power] * df[_timediff]

        _energy = df[_energy_delta].sum(axis=0, skipna=True)
        _time = df[_timestamp].iloc[-1] - df[_timestamp].iloc[0]

        energy.append(_energy)
        time.append(_time)

    return energy, time

def main():
     # argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv_file', required=True, help='path to the csv file for parsing')
    parser.add_argument('--ngpus', default=2, type=int)

    args = parser.parse_args()

    # read csv
    df = pd.read_csv(args.csv_file)

    # get power, total time
    energy, time = parse_for_gpus(df, args.ngpus)

    # print
    # print("Total Energy Consumption: {:.3f}J".format(float(energy)/10**9))
    # print("Total Time: {:.2f}ms".format(float(time)/10**6))
    # print("Peak GPU Util: {}%".format(peak_gpu_util))
    # print("Peak Power: {:.1f}W".format(peak_power))
    # print("Avg Power: {:.2f}W".format(energy/time))
    for i in range(args.ngpus):
        print("GPU {} ---".format(i))
        print("Total Energy Consumption: {:.3f}J".format(float(energy[i])/ 10**9))
        print("Total Time: {:.2f}ms".format(float(time[i]) / 10**6))
        print("Avg Power: {:.2f}W".format(energy[i]/time[i]))

if __name__ == '__main__':
    main()
