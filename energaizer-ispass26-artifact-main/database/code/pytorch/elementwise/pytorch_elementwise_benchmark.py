import os
import numpy as np
import argparse

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

# NVML
try:
    import pynvml
except:
    pass

import time
import math

# https://github.com/ml-energy/zeus/blob/master/zeus/monitor/power.py#L47
#### NVML polling function ####
def poll_nvml(gpu_list, save_to, update_period=0.01, poll_clock=False):
    try:
        pynvml.nvmlInit()
        with open(save_to, "a", buffering=1) as f:
            while True:
                stats = []
                for rank in gpu_list:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
                    # power
                    metric = pynvml.nvmlDeviceGetFieldValues(handle, [pynvml.NVML_FI_DEV_POWER_INSTANT])[0]
                    stats.append(metric.value.uiVal)
                    # sm clock freq
                    if poll_clock:
                        freq = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                        stats.append(freq)
                # stats = pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
                stats_str = ",".join(map(lambda p: str(p / 1000), stats))
                now = time.time()
                f.write(f"{now},{stats_str}\n")
                if (sleep_time := update_period - (time.time() - now)) > 0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        return
    
def get_time_per_iter(op, inputs):
    with torch.no_grad():
        if op  == 'pointwise_mul':
            for i in range(100):
                _ = inputs[0] * inputs[1]
                torch.cuda.synchronize()
        elif op == 'pointwise_add':
            for i in range(100):
                _ = inputs[0] + inputs[1]
                torch.cuda.synchronize()
        elif op == 'scalar_mul':
            for i in range(100):
                _ = inputs[0] * inputs[1]
                torch.cuda.synchronize()
        elif op == 'scalar_add':
            for i in range(100):
                _ = inputs[0] + inputs[1]
                torch.cuda.synchronize()
        elif op == 'typecast_from_bf16_to_fp32':
            for i in range(100):
                _ = inputs.to(dtype=torch.float32)
                torch.cuda.synchronize()
        elif op == 'typecast_from_fp32_to_bf16':
            for i in range(100):
                _ = inputs.to(dtype=torch.bfloat16)
                torch.cuda.synchronize()
        else:
            raise NotImplementedError()

    with torch.no_grad():
        if op == 'pointwise_mul':
            start_time = time.time()
            _ = inputs[0] * inputs[1]
            torch.cuda.synchronize()
            end_time = time.time()
        elif op == 'pointwise_add':
            start_time = time.time()
            _ = inputs[0] + inputs[1]
            torch.cuda.synchronize()
            end_time = time.time()
        elif op == 'scalar_mul':
            start_time = time.time()
            _ = inputs[0] * inputs[1]
            torch.cuda.synchronize()
            end_time = time.time()
        elif op == 'scalar_add':
            start_time = time.time()
            _ = inputs[0] + inputs[1]
            torch.cuda.synchronize()
            end_time = time.time()
        elif op == 'typecast_from_bf16_to_fp32':
            start_time = time.time()
            _ = inputs.to(dtype=torch.float32)
            torch.cuda.synchronize()
            end_time = time.time()
        elif op == 'typecast_from_fp32_to_bf16':
            start_time = time.time()
            _ = inputs.to(dtype=torch.bfloat16)
            torch.cuda.synchronize()
            end_time = time.time()
        else:
            raise NotImplementedError
    
    return (end_time - start_time)

def run_iterations(op, inputs, repeats):
    with torch.no_grad():
        if op  == 'pointwise_mul':
            for i in range(repeats):
                _ = inputs[0] * inputs[1]
                torch.cuda.synchronize()
        elif op == 'pointwise_add':
            for i in range(repeats):
                _ = inputs[0] + inputs[1]
                torch.cuda.synchronize()
        elif op == 'scalar_mul':
            for i in range(repeats):
                _ = inputs[0] * inputs[1]
                torch.cuda.synchronize()
        elif op == 'scalar_add':
            for i in range(repeats):
                _ = inputs[0] + inputs[1]
                torch.cuda.synchronize()
        elif op == 'typecast_from_bf16_to_fp32':
            for i in range(repeats):
                _ = inputs.to(dtype=torch.float32)
                torch.cuda.synchronize()
        elif op == 'typecast_from_fp32_to_bf16':
            for i in range(repeats):
                _ = inputs.to(dtype=torch.bfloat16)
                torch.cuda.synchronize()
        else:
            raise NotImplementedError()

def main():

    parser = argparse.ArgumentParser()

    # NVML
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--iterations', default=1000, type=int)
    parser.add_argument('--nvml_save_to', default='tmp', type=str)
    parser.add_argument('--nvml_update_period', type=float, default=0.01, help='nvml update interval in s')
    parser.add_argument('--nvml_poll_clock', default=False, action='store_true')

    # Precision
    parser.add_argument('--precision', type=str, default='bf16', choices=['fp16', 'fp32', 'bf16'])
    
    # Workload dimension
    parser.add_argument('--workload_dim', type=int, nargs='*', required=True)

    # Workload type
    parser.add_argument('--workload_op', type=str, help='specific function within the type (e.g., pointwise_mul)', required=True)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # dtype = torch.float16 if args.precision == 'fp16' else torch.float32
    if args.precision == 'fp16':
        dtype = torch.float16
    elif args.precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32 

    input_tensor = torch.rand(args.workload_dim, device=device, dtype=dtype)
    if (args.workload_op == 'pointwise_mul') or (args.workload_op == 'pointwise_add'):
        inputs = (input_tensor, input_tensor)
    elif (args.workload_op == 'scalar_mul') or (args.workload_op == 'scalar_add'):
        random_number = torch.rand(1, dtype=dtype, device=device)
        inputs = (input_tensor, random_number)
    else:
        inputs = input_tensor
    
    if args.iterations < 0:
        time_per_iter = get_time_per_iter(args.workload_op, inputs)
        iterations = math.ceil(30.0 / time_per_iter)
    else:
        iterations = args.iterations
    
    nvml_save_to = args.nvml_save_to + '_iter{}.csv'.format(iterations)

    # Single GPU
    size = 1
    pynvml.nvmlInit()
    with open(nvml_save_to, 'w') as f:
        header = ['timestamp']
        for rank in range(size):
            header.append('power_{}'.format(rank))
            if args.nvml_poll_clock:
                header.append('sm_clock_{}'.format(rank))
        header_str = ','.join(str(x) for x in header)
        header_str += '\n'
        f.write(header_str)
        # print(header_str)
        f.close()

    mp.set_start_method("spawn")
 
    # p1 is NVML
    p1 = mp.Process(target=poll_nvml, args=([args.cuda_device], nvml_save_to, args.nvml_update_period, args.nvml_poll_clock,))
    p1.start()

    # p2 is torch
    p2 = mp.Process(target=run_iterations, args=(args.workload_op, inputs, iterations))
    p2.start()
    
    p2.join()
    p1.terminate()
    p1.join()

    pynvml.nvmlShutdown()

if __name__ == '__main__':
    main()
