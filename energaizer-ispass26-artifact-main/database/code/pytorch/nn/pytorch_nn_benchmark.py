import os
import numpy as np
import argparse

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from torch.nn.attention import SDPBackend, sdpa_kernel

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
    
# nn.Conv2d / Conv1d
def run_conv(inputs, output_channel_dim, kernel_size, stride, padding, function, device):
    if function.lower() == 'conv2d':
        m = nn.Conv2d(inputs.shape[1], output_channel_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, dtype=inputs.dtype)
    elif function.lower() == 'conv1d':
        m = nn.Conv1d(inputs.shape[1], output_channel_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, dtype=inputs.dtype)
    else:
        raise NotImplementedError()

    m.to(device)
    
    return m
    
# Pointwise functions: ReLU, GELU, Tanh, Sigmoid
def run_pointwise(inputs, function, device):
    if function.lower() == 'relu':
        m = nn.ReLU()
    elif function.lower() == 'gelu':
        m = nn.GELU()
    elif function.lower() == 'tanh':
        m = nn.Tanh()
    elif function.lower() == 'sigmoid':
        m = nn.Sigmoid()
    elif function.lower() == 'silu':
        m = nn.SiLU()
    else:
        raise NotImplementedError()
    
    m.to(device)

    return m

# Reduction and nomralizations: Softmax, LayerNorm
def run_nonlinear(inputs, function, device):
    if function.lower() == 'softmax':
        m = nn.Softmax(dim=-1)
    elif function.lower() == 'layernorm':
        m = nn.LayerNorm(inputs.shape[1:], dtype=inputs.dtype) # the first dimension in inputs should be the batch information
    else:
        raise NotImplementedError()
    
    m.to(device)

    return m

def run_iterations(module, inputs, repeats):
    with torch.no_grad():
        for i in range(repeats):
            _ = module(inputs)
            torch.cuda.synchronize()

def get_time_per_iter(module, inputs):
    # warmup
    with torch.no_grad():
        for i in range(10):
            _ = module(inputs)
            torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        _ = module(inputs)
        torch.cuda.synchronize()
    end_time = time.time()

    return (end_time - start_time)

def main():
    
    parser = argparse.ArgumentParser()

    # NVML
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--iterations', default=1000, type=int)
    parser.add_argument('--nvml_save_to', default='tmp', type=str)
    parser.add_argument('--nvml_update_period', type=float, default=0.01, help='nvml update interval in s')
    parser.add_argument('--nvml_poll_clock', default=False, action='store_true')

    # Precision
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'fp32', 'bf16'])
    
    # Workload dimension
    parser.add_argument('--workload_dim', type=int, nargs='*', required=True)

    # Workload type
    parser.add_argument('--workload_type', type=str, choices=['conv', 'pointwise', 'nonlinear'], required=True)
    parser.add_argument('--workload_function', type=str, help='specific function within the type (e.g., conv2d)', required=True)

    # For convolutions, need to specify more
    parser.add_argument('--conv_output_channel_dim', type=int, default=128)
    parser.add_argument('--conv_kernel_size', type=int, default=3)
    parser.add_argument('--conv_stride', type=int, default=1)
    parser.add_argument('--conv_padding', type=int, default=1)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # dtype = torch.float16 if args.precision == 'fp16' else torch.float32
    if args.precision == 'fp16':
        dtype = torch.float16
    elif args.precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32 

    # Construct input
    input_tensor = torch.rand(args.workload_dim, device=device, dtype=dtype)

    if args.workload_type == 'conv':
        m = run_conv(input_tensor, args.conv_output_channel_dim, \
                     args.conv_kernel_size, args.conv_stride, args.conv_padding, args.workload_function, device)
    elif args.workload_type == 'pointwise':
        m = run_pointwise(input_tensor, args.workload_function, device)
    elif args.workload_type == 'nonlinear':
        m = run_nonlinear(input_tensor, args.workload_function, device)
    else:
        raise NotImplementedError()
    
    if args.iterations < 0:
        time_per_iter = get_time_per_iter(m, input_tensor)
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
    p2 = mp.Process(target=run_iterations, args=(m, input_tensor, iterations))
    p2.start()
    
    p2.join()
    p1.terminate()
    p1.join()

    pynvml.nvmlShutdown()

if __name__ == '__main__':
    main()

