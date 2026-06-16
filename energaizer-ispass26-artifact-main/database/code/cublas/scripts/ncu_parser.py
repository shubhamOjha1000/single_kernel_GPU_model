import os
import csv
import argparse

KERNELS_EXCLUDED=['scal_kernel', 'initialize', 'curand', 'scal_64addr_kernel']
# KERNELS_EXCLUDED = []

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_folder', required=True, help='path to the folder that has all csv files to be parsed')
    parser.add_argument('--save_to', required=True, help='filename you want to save the summary result')

    args = parser.parse_args()

    # Name of csv file should match that for the uBench (power/NVML) one
    # see cuda-kernel-benchmark/python-scripts/parse_folder.py

    print("[INFO] The csv files in the folder should be names as follows: ")
    print("[INFO] [gemm]_[dimM]_[dimN]_[dimK]_[transpose]_[precM]_[precA]_[useTensorCore]_[initOption].csv")
    print("[INFO] If there exists any file that doesn't have this name format")
    print("[INFO] this script will skip that file.")

    # create a list of python dicts
    summary = []
    """
    fieldnames = ['dimM', 'dimN', 'dimK', \
                  'precM' ,'precA', 'useTensorCore', \
                  'kernel_name', 'grid_size', 'block_size']
    # """
    
    # iterate through all files in this folder
    for subdir, _, files in os.walk(args.path_to_folder):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith('.csv'):
                print("Found a csv file %s", filepath)
                filename = str(file).replace(".csv", "") # remove .csv from the filename
                fields = filename.split('_')
                if ((len(fields) != 9) and (len(fields) != 10)):
                    print("[WARN] This file name doesn't meet the requirement. Skipping this file.")
                    continue
                else:
                    if (len(fields) == 9):
                        temp = {'gemm': fields[0], \
                                'batch': 1, \
                                'dimM': fields[1], \
                                'dimN': fields[2], \
                                'dimK': fields[3], \
                                'trans': fields[4], \
                                'precM': fields[5], \
                                'precA': fields[6], \
                                'useTensorCore': fields[7]}
                    elif (len(fields) == 10):
                        temp = {'gemm': fields[0], \
                                'batch': fields[1],
                                'dimM': fields[2], \
                                'dimN': fields[3], \
                                'dimK': fields[4], \
                                'trans': fields[5], \
                                'precM': fields[6], \
                                'precA': fields[7], \
                                'useTensorCore': fields[8]}

                # Read the content - we have to check "Kernel Name", "Block Size" "Grid Size", and all stats in "Launch Statistics"
                fields = None
                kernel_name = None
                block_size = None
                grid_size = None
                launch_stats = {}

                kernel_idx = -1
                block_idx = -1
                grid_idx = -1
                section_idx = -1
                value_idx = -1
                metric_name_idx = -1
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
                        if vals[0] == '"ID"' and fields is None:
                            fields = vals
                            # print(fields)
                            kernel_idx = fields.index('Kernel Name"')
                            block_idx = fields.index('Block Size"')
                            grid_idx = fields.index('Grid Size"')
                            section_idx = fields.index('Section Name"')
                            value_idx = fields.index('Metric Value"')
                            metric_name_idx = fields.index('Metric Name"')
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
                            if block_size is None:
                                block_size = vals[block_idx]
                            if grid_size is None:
                                grid_size = vals[grid_idx]

                            # print(vals[section_idx])
                            if vals[section_idx] == 'Launch Statistics"' and vals[metric_name_idx] not in launch_stats.keys():
                                launch_stats[vals[metric_name_idx]] = vals[value_idx]
                            if vals[section_idx] == 'GPU Speed Of Light Throughput"' and vals[metric_name_idx] not in launch_stats.keys():
                                # print(vals)
                                launch_stats[vals[metric_name_idx]] = vals[value_idx]
                            if vals[section_idx] == 'Occupancy"' and vals[metric_name_idx] not in launch_stats.keys():
                                launch_stats[vals[metric_name_idx]] = vals[value_idx]
                if kernel_name is None:
                    continue

                temp['kernel_name'] = kernel_name[:-1]
                temp['grid_size'] = grid_size[:-1]
                temp['block_size'] = block_size[:-1]

                # print(temp)
                # exit()
                for key, value in launch_stats.items():
                    temp[key[:-1]] = value.replace('",', '').replace(' ', '').replace('\n', '')

                temp['max_concurrent_block'] = min(int(temp['Block Limit SM']), int(temp['Block Limit Registers']), \
                                                   int(temp['Block Limit Shared Mem']), int(temp['Block Limit Warps']))
                
                summary.append(temp)

    with open(args.save_to, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
                        
if __name__ == '__main__':
    main()
