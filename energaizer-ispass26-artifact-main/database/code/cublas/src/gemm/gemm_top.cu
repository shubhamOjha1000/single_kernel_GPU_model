// Code from https://github.com/hibagus/CUDA_Bench/blob/main/src/gemm/gemm_cublas.cu

#include <gemm/gemm_top.cuh>
#include <gemm/gemm_cublas.cuh>
#include <gemm/gemm_global.cuh>
#include <gemm/strided_batched_gemm_cublas.cuh>
#include <util/gpucheck.cuh>
#include <util/gpuinfo.cuh>
#include <util/precision_select.cuh>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <iostream>

#include "nvmlClass.h"

float gemm_cublas(std::string const filename, int dev){
    // Detect Available CUDA Devices
    int nDevices;
    gpuErrchk(cudaGetDeviceCount(&nDevices));
    print_cuda_device_info(nDevices);
    if(nDevices>0) {std::cout << "[WARN] This program does not currently support Multi-GPU run.\n";}

    // Create nvml
    cudaSetDevice(dev);
    std::cout << "[Setting] Using GPU " << dev << "\n";

    // std::string const filename = { "data/gpuStats.csv" };
    nvmlClass nvml( dev, filename );

    float times = 0;

    // Select precision
    if (gmulprecision == PRECISION_FP64 && gaccprecision == PRECISION_FP64){
        if (gbatch > 1){
            times = gemmstridedbatched_cublas_kernel_double_double_double(nvml);
        }
        else{
            times = gemm_cublas_kernel_double_double_double(nvml);
        }
    }

    else if (gmulprecision == PRECISION_FP32 && gaccprecision == PRECISION_FP32){
        
            if (gbatch > 1){
                times = gemmstridedbatched_cublas_kernel_float_float_float(nvml);
            }
            else{
                times = gemm_cublas_kernel_float_float_float(nvml);
            }
    }

    else if (gmulprecision == PRECISION_FP16 && gaccprecision == PRECISION_FP32){
        if (gbatch > 1){
            times = gemmstridedbatched_cublas_kernel_float_half_float(nvml);
        }
        else{
            times = gemm_cublas_kernel_float_half_float(nvml);
        }
    }

    else if (gmulprecision == PRECISION_FP16 && gaccprecision == PRECISION_FP16){

        if (gbatch > 1){
            if (ghalf_prec_accumulate_full){
                times = gemmstridedbatched_cublas_kernel_float_half_half(nvml);
            }
            else{
                times = gemmstridedbatched_cublas_kernel_half_half_half(nvml);
            }
        }
        else {
            if (ghalf_prec_accumulate_full){
                times = gemm_cublas_kernel_float_half_half(nvml);
            }
            else{
                times = gemm_cublas_kernel_half_half_half(nvml);
            }
        }
    }

    else if (gmulprecision == PRECISION_BF16 && gaccprecision == PRECISION_FP32){
        // GemmEx BF16 A/B and FP32 Result --> alpha, beta should be float, a/b should be bf16, ctype should be float
        // --> S = float, M = bf16, A = float
        if (gbatch > 1){
            times = gemmstridedbatched_cublas_kernel_float_bf16_float( nvml);
        }
        else{
            times = gemm_cublas_kernel_float_bf16_float(nvml);
        }
    }

    else if (gmulprecision == PRECISION_BF16 && gaccprecision == PRECISION_BF16){

        // BF16 M, BF16 A --> Still S should be Float
        // https://docs.nvidia.com/cuda/cublas/#cublasgemmex
        if (gbatch > 1){
            times = gemmstridedbatched_cublas_kernel_float_bf16_bf16(nvml);
        }
        else {
            times = gemm_cublas_kernel_float_bf16_bf16(nvml);
        }
    }

    else if (gmulprecision == PRECISION_INT8 && gaccprecision == PRECISION_INT8){
        if (gbatch > 1){
            times = gemmstridedbatched_cublas_kernel_int_int8_int(nvml);
        }
        else{
            times = gemm_cublas_kernel_int_int8_int(nvml);
        }
    }

    return times;
}
