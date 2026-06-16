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

from typing import Union, Callable, Optional

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
    
def run_model(repeats, attention_type, query, key, value):
    # Attention Type
    if attention_type == 'math':
        _attention_type = SDPBackend.MATH
    elif attention_type == 'flash_attention':
        _attention_type = SDPBackend.FLASH_ATTENTION
    elif attention_type == 'efficient_attention':
        _attention_type = SDPBackend.EFFICIENT_ATTENTION
    elif attention_type == 'default':
        pass
    else:
        raise NotImplementedError()
    
    if attention_type != 'default':
        with sdpa_kernel(_attention_type):
            with torch.no_grad():
                for i in range(repeats):
                    _ = F.scaled_dot_product_attention(query, key, value)
                    torch.cuda.synchronize()

    else:
        with torch.no_grad():
            for i in range(repeats):
                _ = F.scaled_dot_product_attention(query, key, value)
                torch.cuda.synchronize()

def get_time_per_iter(attention_type, query, key, value):
    # Attention Type
    if attention_type == 'math':
        _attention_type = SDPBackend.MATH
    elif attention_type == 'flash_attention':
        _attention_type = SDPBackend.FLASH_ATTENTION
    elif attention_type == 'efficient_attention':
        _attention_type = SDPBackend.EFFICIENT_ATTENTION
    elif attention_type == 'default':
        pass
    else:
        raise NotImplementedError()
    
    if attention_type != 'default':
        with sdpa_kernel(_attention_type):
            with torch.no_grad():
                # warmup
                for i in range(10):
                    _ = F.scaled_dot_product_attention(query, key, value)
                    torch.cuda.synchronize()
            
            start_time = time.time()
            with torch.no_grad():
                _ = F.scaled_dot_product_attention(query, key, value)
                torch.cuda.synchronize()
            end_time = time.time()

            return (end_time - start_time)

    else:
        with torch.no_grad():
            # warmup
            for i in range(10):
                _ = F.scaled_dot_product_attention(query, key, value)
                torch.cuda.synchronize()

        start_time = time.time()
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(query, key, value)
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

    # Backend options
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'fp32', 'bf16'])
    parser.add_argument('--sdpa_backend', choices=['math', 'flash_attention', 'efficient_attention', 'default'], default='default')

    # SDPA options
    parser.add_argument('--sdpa_batch_size', type=int, default=32)
    parser.add_argument('--sdpa_num_heads', type=int, default=32)
    parser.add_argument('--sdpa_q_sequence_len', type=int, default=-1)
    parser.add_argument('--sdpa_sequence_len', type=int, default=1024)
    parser.add_argument('--sdpa_embed_dimension', type=int, default=32)
    parser.add_argument('--sdpa_decoding_phase', default=False, action='store_true')
    parser.add_argument('--sdpa_gqa', default=False, action='store_true')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # dtype = torch.float16 if args.precision == 'fp16' else torch.float32
    if args.precision == 'fp16':
        dtype = torch.float16
    elif args.precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32 

    # During decoding (inference/token generation) phase, the sequence length for query becomes 1 (one token at a time)
    query_seq_len = 1 if args.sdpa_decoding_phase else (args.sdpa_q_sequence_len if args.sdpa_q_sequence_len > 0 else args.sdpa_sequence_len)
    num_heads = args.sdpa_num_heads

    query = torch.rand(args.sdpa_batch_size, num_heads, query_seq_len, args.sdpa_embed_dimension, device=device, dtype=dtype)
    key = torch.rand(args.sdpa_batch_size, num_heads, args.sdpa_sequence_len, args.sdpa_embed_dimension, device=device, dtype=dtype)
    value = torch.rand(args.sdpa_batch_size, num_heads, args.sdpa_sequence_len, args.sdpa_embed_dimension, device=device, dtype=dtype)

    if args.iterations < 0:
        time_per_iter = get_time_per_iter(args.sdpa_backend, query, key, value)
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

    # p2 is Torch
    p2 = mp.Process(target=run_model, args=(iterations, args.sdpa_backend, query, key, value))
    p2.start()

    p2.join()

    p1.terminate()
    p1.join()

    pynvml.nvmlShutdown()

if __name__ == '__main__':
    main()
