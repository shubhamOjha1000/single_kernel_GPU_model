// Code from https://github.com/hibagus/CUDA_Bench/blob/main/src/gemm/gemm_cublas.cu

#include <gemm/gemm_cublas.cuh>
#include <gemm/gemm_global.cuh>
#include <util/gpucheck.cuh>
#include <util/gpuinfo.cuh>
#include <util/precision_select.cuh>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>

#include "nvmlClass.h"

#include <curand.h>

// #include <l2flush/benchmark.cuh>

void GPU_fill_rand(float *A, int row, int col)
{
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);

    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    curandGenerateUniform(prng, A, row*col);
}

float gemm_cublas_kernel_double_double_double(nvmlClass &nvml){
    return gemm_cublas_kernel<double, double, double>(nvml);
}

float gemm_cublas_kernel_float_float_float(nvmlClass &nvml){
    return gemm_cublas_kernel<float, float, float>(nvml);
}

float gemm_cublas_kernel_float_half_float(nvmlClass &nvml){
    return gemm_cublas_kernel<float, half, float>(nvml);
}

float gemm_cublas_kernel_float_half_half(nvmlClass &nvml){
    return gemm_cublas_kernel<float, half, half>(nvml);
}

float gemm_cublas_kernel_half_half_half(nvmlClass &nvml){
    return gemm_cublas_kernel<half, half, half>(nvml);
}

float gemm_cublas_kernel_float_bf16_float(nvmlClass &nvml){
    return gemm_cublas_kernel<float, __nv_bfloat16, float>(nvml);
}

float gemm_cublas_kernel_float_bf16_bf16(nvmlClass &nvml){
    return gemm_cublas_kernel<float, __nv_bfloat16, __nv_bfloat16>(nvml);
}

float gemm_cublas_kernel_int_int8_int(nvmlClass &nvml){
    return gemm_cublas_kernel<int, int8_t, int>(nvml);
}

