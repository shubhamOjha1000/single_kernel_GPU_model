#!/bin/bash

# Test: Generate final database csv file from the **parsed** results

python3 ../generate_lut.py --sdpa_result_folder ../../../../kubelut/flashattn_0323_diffq_yz8 --profile_ncu --save_to test_a100_flashattn_lut.csv