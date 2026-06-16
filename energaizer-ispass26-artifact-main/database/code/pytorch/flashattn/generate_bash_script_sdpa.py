import os
import sys
import argparse

import pandas as pd

import yaml

def main():
    parser = argparse.ArgumentParser()

    # NVML
    parser.add_argument('--profile_nvml', default=False, action='store_true')
    parser.add_argument('--nvml_save_path', type=str)

    # Workload CSV / Bash path
    parser.add_argument('--workload_csv_path', type=str, required=True)
    parser.add_argument('--bash_save_path', type=str, required=True)

    # NCU 
    parser.add_argument('--profile_ncu', default=False, action='store_true')
    parser.add_argument('--ncu_save_path', type=str)
    parser.add_argument('--ncu_bin_path', default='/usr/local/cuda/bin/ncu', type=str)
    parser.add_argument('--ncu_metrics', default=None, nargs='*')

    # Python Command
    parser.add_argument('--python_bin', type=str, default='~/miniconda3/envs/pytorch/bin/python3', required=False)

    # GPU Setting
    parser.add_argument('--cuda_device', default=0, type=int, help='Single GPU only')
    parser.add_argument('--lock_gpu_clock', default=False, action='store_true')
    parser.add_argument('--gpu_clock_freq', default=1410, type=int, help='gpu clock frequency to be locked in MHz')
    parser.add_argument('--set_power_limit', default=False, action='store_true')
    parser.add_argument('--gpu_power_limit', default=250, type=int)
    parser.add_argument('--sudo_password', type=str)

    args = parser.parse_args()

    workload = pd.read_csv(args.workload_csv_path)

    if os.path.exists(args.bash_save_path):
        os.remove(args.bash_save_path)

    if args.ncu_metrics is not None:
        metrics = []
        for p in args.ncu_metrics:
            with open(p) as f:
                _metrics = yaml.safe_load(f)['metrics']
                metrics.extend(_metrics)

    with open(args.bash_save_path, 'w') as f:

        # Check if nvml folder exists
        if not os.path.exists(args.nvml_save_path):
            # os.mkdir(args.nvml_save_path)
            # print('mkdir {}'.format(args.nvml_save_path))
            f.write('''\
mkdir {}\n
'''.format(args.nvml_save_path))

        # Check if ncu folder exists
        if args.profile_ncu and (not os.path.exists(args.ncu_save_path)):
            # os.mkdir(args.ncu_save_path)
            # print('mkdir {}'.format(args.ncu_save_path))
            f.write('''\
mkdir {}\n
'''.format(args.ncu_save_path))

        # Lock clock / power limit if the option is given
        if args.lock_gpu_clock:
            # os.system('echo {} | sudo -S nvidia-smi -i {} -lgc {},{}'.format(args.sudo_password, args.cuda_device, args.gpu_clock_freq, args.gpu_clock_freq))
            # print('echo {} | sudo -S nvidia-smi -i {} -lgc {},{}'.format(args.sudo_password, args.cuda_device, args.gpu_clock_freq, args.gpu_clock_freq))
            f.write('''\
echo {} | sudo -S nvidia-smi -i {} -lgc {},{}\n
'''.format(args.sudo_password, args.cuda_device, args.gpu_clock_freq, args.gpu_clock_freq))

        if args.set_power_limit:
            # os.system('echo {} | sudo -S nvidia-smi -i {} -pl {}'.format(args.sudo_password, args.cuda_device, args.gpu_power_limit))
            # print('echo {} | sudo -S nvidia-smi -i {} -pl {}'.format(args.sudo_password, args.cuda_device, args.gpu_power_limit))
            f.write('''\
echo {} | sudo -S nvidia-smi -i {} -pl {}\n
'''.format(args.sudo_password, args.cuda_device, args.gpu_power_limit))
        
        # Iterate through workloads
        for _, config in workload.iterrows():

            # Compose python instruction
            cmd = '{} pytorch_sdpa_benchmark.py'.format(args.python_bin)
            # cmd += ' --iterations {}'.format(args.iterations)
            cmd += ' --cuda_device {}'.format(args.cuda_device)
            cmd += ' --precision {}'.format(config['precision'])
            cmd += ' --sdpa_backend {}'.format(config['backend'])
            cmd += ' --sdpa_batch_size {}'.format(config['sdpa_batch_size'])
            cmd += ' --sdpa_num_heads {}'.format(config['sdpa_num_heads'])
            cmd += ' --sdpa_sequence_len {}'.format(config['sdpa_sequence_len'])
            cmd += ' --sdpa_embed_dimension {}'.format(config['sdpa_embed_dimension'])

            if 'sdpa_q_sequence_len' in config.keys():
                cmd += ' --sdpa_q_sequence_len {}'.format(config['sdpa_q_sequence_len'])

            if config['sdpa_decoding_phase']:
                cmd += ' --sdpa_decoding_phase'
            if config['sdpa_gqa']:
                cmd += ' --sdpa_gqa'

            if 'sdpa_q_sequence_len' in config.keys():
                log_save_prefix = 'sdpa_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(config['precision'], config['backend'], \
                                                                           config['sdpa_batch_size'], config['sdpa_num_heads'], \
                                                                           config['sdpa_q_sequence_len'], \
                                                                           config['sdpa_sequence_len'], config['sdpa_embed_dimension'], \
                                                                           'decode' if config['sdpa_decoding_phase'] else 'prefill', \
                                                                           'gqa' if config['sdpa_gqa'] else 'normal')
            else:
                log_save_prefix = 'sdpa_{}_{}_{}_{}_{}_{}_{}_{}'.format(config['precision'], config['backend'], \
                                                                        config['sdpa_batch_size'], config['sdpa_num_heads'], \
                                                                        config['sdpa_sequence_len'], config['sdpa_embed_dimension'], \
                                                                        'decode' if config['sdpa_decoding_phase'] else 'prefill', \
                                                                        'gqa' if config['sdpa_gqa'] else 'normal')

            # If NVML, run the code with nvml
            if args.profile_nvml:
                nvml_cmd = 'CUDA_VISIBLE_DEVICES={} '.format(args.cuda_device) + cmd + ' --nvml_save_to {}'.format(os.path.join(args.nvml_save_path, log_save_prefix)) + ' --nvml_poll_clock' + ' --iterations -1'
                # print(nvml_cmd)
                # os.system(nvml_cmd)
                f.write(nvml_cmd+'\n')

            # If NCU, run the code with ncu
            if args.profile_ncu:
                ncu_log = os.path.join(args.ncu_save_path, 'ncu_' + log_save_prefix + '.csv')
                ncu_cmd = 'echo {} | sudo -S CUDA_VISIBLE_DEVICES={} {} --log-file {} --csv'.format(args.sudo_password, args.cuda_device, args.ncu_bin_path, ncu_log)
                ncu_cmd += ' ' + cmd + ' --iterations 1'
                # print(ncu_cmd)
                # os.system(ncu_cmd)
                f.write(ncu_cmd+'\n')

                # If NCU + metrics, run the code
                if args.ncu_metrics is not None:
                    metrics_list = ''
                    for idx, entry in enumerate(metrics):
                        metrics_list += entry
                        if idx < len(metrics) - 1:
                            metrics_list += ','
                    ncu_metrics_log = os.path.join(args.ncu_save_path, 'metrics_' + log_save_prefix + '.csv')
                    metrics_cmd = 'echo {} | sudo -S CUDA_VISIBLE_DEVICES={} {} --log-file {} --csv --metrics {}'.format(args.sudo_password, args.cuda_device, args.ncu_bin_path, ncu_metrics_log, metrics_list)
                    metrics_cmd += ' ' + cmd + ' --iterations 1'
                    # print(metrics_cmd)
                    # os.system(metrics_cmd)
                    f.write(metrics_cmd+'\n')

        # Exit
        if args.lock_gpu_clock:
            # print('echo {} | sudo -S nvidia-smi -i {} -rgc'.format(args.sudo_password, args.cuda_device))
            # os.system('echo {} | sudo -S nvidia-smi -i {} -rgc'.format(args.sudo_password, args.cuda_device))
            f.write('''\
echo {} | sudo -S nvidia-smi -i {} -rgc\n
'''.format(args.sudo_password, args.cuda_device))
            
        if args.set_power_limit:
            # print('echo {} | sudo -S nvidia-smi -i {} -pl 250'.format(args.sudo_password, args.cuda_device))
            # os.system('echo {} | sudo -S nvidia-smi -i {} -pl 250'.format(args.sudo_password, args.cuda_device))
            f.write('''\
echo {} | sudo -S nvidia-smi -i {} -pl 250\n
'''.format(args.sudo_password, args.cuda_device))

if __name__ == '__main__':
    main()
