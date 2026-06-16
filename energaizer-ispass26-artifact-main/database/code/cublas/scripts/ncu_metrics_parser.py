# NCU metrics only parsing (e.g., memory hierarchy statistics)

import os
import csv
import argparse
import yaml

KERNELS_EXCLUDED=['scal_kernel', 'initialize', 'curand']

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_folder', required=True, help='path to the folder that has all csv files to be parsed')
    parser.add_argument('--save_to', required=True, help='filename you want to save the summary result')
    parser.add_argument('--ncu_metrics', nargs='*', required=True, help='path to the yaml file for metrics profiled')

    args = parser.parse_args()

    if args.ncu_metrics is not None:
        metrics = []
        for p in args.ncu_metrics:
            with open(p) as f:
                _metrics = yaml.safe_load(f)['metrics']
                metrics.extend(_metrics)

    # Name of csv file should match that for the uBench (power/NVML) one

    print("[INFO] The csv files in the folder should be names as follows: ")
    print("[INFO] metrics_[gemm]_[dimM]_[dimN]_[dimK]_[transpose]_[precM]_[precA]_[useTensorCore]_[initOption].csv")
    print("[INFO] If there exists any file that doesn't have this name format")
    print("[INFO] this script will skip that file.")

    # create a list of python dicts
    summary = []
    
    # iterate through all files in this folder
    for subdir, _, files in os.walk(args.path_to_folder):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith('.csv'):
                print("Found a csv file %s", filepath)
                filename = str(file).replace(".csv", "") # remove .csv from the filename
                fields = filename.split('_')

                # We only want to check 'metrics_....csv'
                if fields[0] != 'metrics':
                    print("[WARN] This file name doesn't meet the requirement. For metrics, the filename should start with 'metrics_'. Skipping this file.")
                    continue

                if ((len(fields) != 10) and (len(fields) != 11)):
                    print("[WARN] This file name doesn't meet the requirement. Skipping this file.")
                    continue
                else:
                    if (len(fields) == 10):
                        temp = {'gemm': fields[1], \
                                'batch': 1, \
                                'dimM': fields[2], \
                                'dimN': fields[3], \
                                'dimK': fields[4], \
                                'trans': fields[5], \
                                'precM': fields[6], \
                                'precA': fields[7], \
                                'useTensorCore': fields[8]}
                    elif (len(fields) == 11):
                        temp = {'gemm': fields[1], \
                                'batch': fields[2],
                                'dimM': fields[3], \
                                'dimN': fields[4], \
                                'dimK': fields[5], \
                                'trans': fields[6], \
                                'precM': fields[7], \
                                'precA': fields[8], \
                                'useTensorCore': fields[9]}
                        
                # Start parsing this file
                fields = None
                kernel_name = None

                kernel_idx = -1
                value_idx = -1
                metric_name_idx = -1
                metrics_collected = {}
                with open(filepath, 'r') as f:
                    for line in f:
                        # Skip first few lines with system messages
                        # print(line)
                        vals = line.split(',"') # csv
                        if len(vals) == 1:
                            continue

                        # Headers
                        # print(vals[0], type(vals[0]))
                        # print(vals)

                        # strip these: ", \n
                        vals = [x.replace('"', '').replace('\n', '') for x in vals]
                        if vals[0] == 'ID' and fields is None:
                            fields = vals
                            # print("Header")
                            # print(fields)
                            kernel_idx = fields.index('Kernel Name')
                            value_idx = fields.index('Metric Value')
                            metric_name_idx = fields.index('Metric Name')
                            # print(metric_name_idx, value_idx)
                            # print(kernel_idx, block_idx)
                            continue
                        # Already found headers
                        else:
                            # Check if kernel name for this row is in the exclusion list
                            skip = False
                            for excluded_kernel in KERNELS_EXCLUDED:
                                if excluded_kernel in vals[kernel_idx]:
                                    skip = True
                                    break
                            if skip:
                                continue

                            if kernel_name is None:
                                kernel_name = vals[kernel_idx]

                            # print(vals[metric_name_idx])
                            # print(vals[metric_name_idx], metrics)
                            if vals[metric_name_idx] in metrics:
                                if vals[metric_name_idx] not in metrics_collected.keys():
                                    metrics_collected[vals[metric_name_idx]] = vals[value_idx]

                
                if kernel_name is None:
                    continue

                # print(metrics_collected)
                for key, value in metrics_collected.items():
                    temp[key] = value.replace('",', '').replace(' ', '').replace('\n', '')
                    # print(key, value)
                summary.append(temp)

            # break

    with open(args.save_to, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        # print(summary[0].keys())
        w.writeheader()
        w.writerows(summary)
                        
if __name__ == '__main__':
    main()
                             

