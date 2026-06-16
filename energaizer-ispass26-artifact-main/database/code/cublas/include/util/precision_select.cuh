// Code from https://github.com/hibagus/CUDA_Bench/blob/main/include/CUDA_Bench/util/precision_select.cuh

#pragma once

enum Precision {PRECISION_FP64, 
                PRECISION_FP32,
                PRECISION_TF32,
                PRECISION_FP16,
                PRECISION_BF16,
                PRECISION_INT8,
                PRECISION_INT4,
                PRECISION_INT1};