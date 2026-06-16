#!/bin/bash

# Test: Generate final database csv file from the **parsed** results

python3 ../generate_lut_nonlinear.py --result_folder ../../../../kubelut/a10/nonlinear --operation_name softmax --save_to ./ --save_prefix test_a10
python3 ../generate_lut_conv.py --result_folder ../../../../kubelut/a10/conv --save_to test_a10_conv_lut.csv
python3 ../generate_lut_activation.py --result_folder ../../../../kubelut/a10/pointwise --save_to test_a10_activation_lut.csv