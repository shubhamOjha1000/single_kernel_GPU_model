// References:
// Code from https://github.com/hibagus/CUDA_Bench/blob/main/include/CUDA_Bench/gemm/gemm_cublas_launch_fp.cuh
// Code from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Level-3/gemm/cublas_gemm_example.cu

#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <util/precision_select.cuh>
#include <util/gpucheck.cuh>
#include <util/gemm_util.cuh>
#include <gemm/gemm_global.cuh>
// #include <gemm/sgemm_cublas.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>

#include "nvmlClass.h"

#include <curand.h>

void GPU_fill_rand(float *A, int row, int col);

// Inputs: matrices in CPU context (matA, matB)
// matA : gdim_M x gdim_K
// matB : gdim_N x gdim_K
// matC : gdim_M x gdim_N
template<typename S, typename M, typename A>
float gemm_cublas_kernel(nvmlClass &nvml)
{
    // Result matrix C, pointers and other data
    // std::vector<A> matC(gdim_M * gdim_N);
    M* dev_matA;
    M* dev_matB;
    A* dev_matC;
    S alpha;
    S beta;

    // 1: Create cuBLAS handle
    cublasHandle_t cublasH = NULL;
    gpuErrchk(cublasCreate(&cublasH));

    // 2: Allocate memory & copy
    gpuErrchk(cudaMalloc((void**)&dev_matA, gdim_M * gdim_K * sizeof(M)));
    gpuErrchk(cudaMalloc((void**)&dev_matB, gdim_K * gdim_N * sizeof(M)));
    gpuErrchk(cudaMalloc((void**)&dev_matC, gdim_M * gdim_N * sizeof(A)));

    // Changing Async -> Sync    
    initialize_matrix<A><<<((gdim_M*gdim_N)+512-1)/512,512>>>(dev_matC, gdim_M, gdim_N, 0.0);

    float* rand_matA;
    float* rand_matB;
    gpuErrchk(cudaMalloc((void**)&rand_matA, gdim_M * gdim_K * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&rand_matB, gdim_K * gdim_N * sizeof(float)));
    GPU_fill_rand(rand_matA, gdim_M, gdim_K);
    GPU_fill_rand(rand_matB, gdim_K, gdim_N);

    initialize_random<M><<<((gdim_M*gdim_K)+512-1)/512,512>>>(rand_matA, dev_matA, gdim_M, gdim_K);
    initialize_random<M><<<((gdim_N*gdim_K)+512-1)/512,512>>>(rand_matB, dev_matB, gdim_K, gdim_N);
    gpuErrchk(cudaFree(rand_matA));
    gpuErrchk(cudaFree(rand_matB));

    // Host wait until GPU completes transfer
    gpuErrchk(cudaDeviceSynchronize());

    // 3: cuBLAS compute
    // 3.1: Configure cuBLAS options
    cudaDataType_t mulDataType, accDataType;
    cublasComputeType_t computeType;
    cublasGemmAlgo_t algoType;
    cublasOperation_t matA_op;
    cublasOperation_t matB_op;
    bool integerMult;

    switch(gmulprecision)
    {
        case PRECISION_FP64: {
            mulDataType = CUDA_R_64F;
            alpha = 1.0;
            beta = 0.0;
            integerMult = false;
            break;
        }

        case PRECISION_FP32: {
            mulDataType = CUDA_R_32F;
            alpha = 1.0;
            beta = 0.0;
            integerMult = false;
            break;
        }

        case PRECISION_FP16: {
            mulDataType = CUDA_R_16F;
            alpha = 1.0;
            beta = 0.0;
            integerMult = false;
            break;
        }

        case PRECISION_INT8: {
            mulDataType = CUDA_R_8I;
            alpha = 1;
            beta = 0;
            integerMult = true;
            break;
        }

        case PRECISION_BF16: {
            mulDataType = CUDA_R_16BF;
            alpha = 1.0;
            beta = 0.0;
            integerMult = false;
            break;
        }
    }

    switch(gaccprecision)
    {
        case PRECISION_FP64: {
            accDataType = CUDA_R_64F;
            break;
        }

        case PRECISION_FP32: {
            accDataType = CUDA_R_32F;
            break;
        }

        case PRECISION_FP16: {
            accDataType = CUDA_R_16F;
            break;
        }

        case PRECISION_INT8: {
            accDataType = CUDA_R_32I;
            break;
        }

        case PRECISION_BF16: {
            accDataType = CUDA_R_16BF;
            break;
        }
    }

    if (mulDataType==CUDA_R_64F && accDataType==CUDA_R_64F) {
        if (gtensor_cores)
        {
            // verified
            computeType = CUBLAS_COMPUTE_64F;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP; // will fallback to CUBLAS_GEMM_DEFAULT
            // matA_op     = CUBLAS_OP_N;
            // matB_op     = CUBLAS_OP_N; 
        }
        else             
        {
            // verified
            computeType =CUBLAS_COMPUTE_64F_PEDANTIC;          
            algoType    =CUBLAS_GEMM_DEFAULT; 
            // matA_op     = CUBLAS_OP_N;
            // matB_op     = CUBLAS_OP_N;   
        }
    }
    else if (mulDataType==CUDA_R_32F && accDataType==CUDA_R_32F)
    {
        if(gtensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F_FAST_TF32 ; 
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            // matA_op     = CUBLAS_OP_N;
            // matB_op     = CUBLAS_OP_N;
            std::cout << "[WARN] Currently Tensor Cores are not supporting FP32 multiplication and accumulation, and thus lossy precision is used\n";
        }
        else             
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT; 
            // matA_op     = CUBLAS_OP_N;
            // matB_op     = CUBLAS_OP_N;      
        }
    }
    else if ((mulDataType==CUDA_R_16F) && accDataType==CUDA_R_32F)
    {
        if(gtensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            // matA_op     = CUBLAS_OP_N;
            // matB_op     = CUBLAS_OP_N;
        }    
        else             
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT;
            // matA_op     = CUBLAS_OP_N;
            // matB_op     = CUBLAS_OP_N;          
        }
    }
    else if (mulDataType==CUDA_R_16F && accDataType==CUDA_R_16F)
    {
        if(gtensor_cores) 
        {
            if(ghalf_prec_accumulate_full){
                // verified
                computeType = CUBLAS_COMPUTE_32F;          
                algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                // matA_op     = CUBLAS_OP_N;
                // matB_op     = CUBLAS_OP_N;
            }
            else{
                computeType =  CUBLAS_COMPUTE_16F;    
                algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            }
        }
        else             
        {   
            // verified
            computeType = CUBLAS_COMPUTE_16F_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT;
            // matA_op     = CUBLAS_OP_N;
            // matB_op     = CUBLAS_OP_N;     
        }
    }

    else if ((mulDataType==CUDA_R_16BF) && accDataType==CUDA_R_32F)
    {
        if(gtensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            // matA_op     = CUBLAS_OP_N;
            // matB_op     = CUBLAS_OP_N;
        }    
        else             
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT;
            // matA_op     = CUBLAS_OP_N;
            // matB_op     = CUBLAS_OP_N;          
        }
    }
    else if (mulDataType==CUDA_R_16BF && accDataType==CUDA_R_16BF)
    {
        if(gtensor_cores) 
        {
            // verified
            computeType = CUBLAS_COMPUTE_32F;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            // matA_op     = CUBLAS_OP_N;
            // matB_op     = CUBLAS_OP_N;
        }
        else             
        {   
            // verified
            computeType = CUBLAS_COMPUTE_32F_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT;
            // matA_op     = CUBLAS_OP_N;
            // matB_op     = CUBLAS_OP_N;     
        }
    }
    
    else if (mulDataType == CUDA_R_8I && accDataType==CUDA_R_32I)
    {
        std::cout << "[WARN] Promoting accumulation precision to int32 to maintain compability\n";  
        if(gtensor_cores) 
        {
            computeType = CUBLAS_COMPUTE_32I;          
            algoType    = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            // matA_op     = CUBLAS_OP_N;
            // matB_op     = CUBLAS_OP_N;
        }
        else             
        {
           
            computeType = CUBLAS_COMPUTE_32I_PEDANTIC;          
            algoType    = CUBLAS_GEMM_DEFAULT; 
            // matA_op     = CUBLAS_OP_N;
            // matB_op     = CUBLAS_OP_N;
        }
    }
    else
    {
        std::cerr <<"[ERR!] Precision combination is not supported\n\n\n";
        std::exit(1);
    } 

    // Transpose?
    int ldA;
    int ldB;
    if (gtrn_matA) {
        matA_op = CUBLAS_OP_T;
        ldA = gdim_K;
    }
    else {
        matA_op = CUBLAS_OP_N;
        ldA = gdim_M;
    }
    if (gtrn_matB) {
        matB_op = CUBLAS_OP_T;
        ldB = gdim_N;
    }
    else {
        matB_op = CUBLAS_OP_N;
        ldB = gdim_K;
    }
   
    // warmup for a short iters.. (5 here)
    if (gnum_iter > 1) { 
        for (int warmup=0;warmup<5;warmup++)
        {
            gpuErrchk(cublasGemmEx(cublasH,                       // handle to cuBLAS library context
                            matA_op,                      // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                            matB_op,                      // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                            gdim_M,                       // dimension M 
                            gdim_N,                       // dimension N
                            gdim_K,                       // dimension K
                            &alpha,                       // Scaling factor alpha where (alpha)x(AxB)
                            dev_matA,                     // Pointer to Matrix A on Device
                            mulDataType,                  // Data type of Matrix A
                            ldA,                        // Leading Dimension of Matrix A
                            dev_matB,                     // Pointer to Matrix B on Device
                            mulDataType,                  // Data Type of Matrix B
                            ldB,                        // Leading Dimension of Matrix B
                            &beta,                        // Scaling factor beta where (beta)xC
                            dev_matC,                     // Pointer to Matrix C on Device
                            accDataType,                  // Data Type of Matrix C
                            gdim_M,                        // Leading Dimension of Matrix C
                            computeType,                  // Computation Type
                            algoType                      // Computation Algorithm
            ));
        }
    }

    gpuErrchk(cudaDeviceSynchronize());
    
    std::vector<float> elapsed_time_ms(gnum_iter, 0);

    cudaEvent_t time_start, time_stop;
    cudaEventCreate(&time_start);
    cudaEventCreate(&time_stop);

    // 3.2: Launch NVML thread
    std::thread threadStart( &nvmlClass::getStats,
                             &nvml );  // threadStart starts running


    // 3.3: cuBLAS
    for(int iter=0;iter<gnum_iter;iter++)
    {
        
        gpuErrchk(cudaEventRecord(time_start));
        
        gpuErrchk(cublasGemmEx(cublasH,                       // handle to cuBLAS library context
                            matA_op,                      // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                            matB_op,                      // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
                            gdim_M,                       // dimension M 
                            gdim_N,                       // dimension N
                            gdim_K,                       // dimension K
                            &alpha,                       // Scaling factor alpha where (alpha)x(AxB)
                            dev_matA,                     // Pointer to Matrix A on Device
                            mulDataType,                  // Data type of Matrix A
                            ldA,                        // Leading Dimension of Matrix A
                            dev_matB,                     // Pointer to Matrix B on Device
                            mulDataType,                  // Data Type of Matrix B
                            ldB,                        // Leading Dimension of Matrix B
                            &beta,                        // Scaling factor beta where (beta)xC
                            dev_matC,                     // Pointer to Matrix C on Device
                            accDataType,                  // Data Type of Matrix C
                            gdim_M,                        // Leading Dimension of Matrix C
                            computeType,                  // Computation Type
                            algoType                      // Computation Algorithm
        ));

        gpuErrchk(cudaEventRecord(time_stop));
        gpuErrchk(cudaEventSynchronize(time_stop));
        gpuErrchk(cudaEventElapsedTime(&elapsed_time_ms[iter], time_start, time_stop));
        
    }

    /// Wait until cuBLAS is completed
    gpuErrchk(cudaDeviceSynchronize());

    // 3.4: Terminate NVML thread
    std::thread threadKill( &nvmlClass::killThread, &nvml );
    threadStart.join( );
    threadKill.join( );

    float avg_time_elapsed_ms = 0;
    for (int j=0; j<gnum_iter; j++){
        avg_time_elapsed_ms += elapsed_time_ms[j];
    }
    avg_time_elapsed_ms = avg_time_elapsed_ms / gnum_iter;
    std::printf("Average Time per Iteration: %f\n", avg_time_elapsed_ms);


    // 5: Clear
    gpuErrchk(cudaFree(dev_matA));
    gpuErrchk(cudaFree(dev_matB));
    gpuErrchk(cudaFree(dev_matC));
    gpuErrchk(cublasDestroy(cublasH));
    gpuErrchk(cudaEventDestroy(time_start));
    gpuErrchk(cudaEventDestroy(time_stop));
    gpuErrchk(cudaDeviceReset());

    return avg_time_elapsed_ms;

}

float gemm_cublas_kernel_double_double_double(nvmlClass &nvml);
float gemm_cublas_kernel_float_float_float(nvmlClass &nvml);
float gemm_cublas_kernel_float_half_float(nvmlClass &nvml);
float gemm_cublas_kernel_half_half_half(nvmlClass &nvml);
float gemm_cublas_kernel_float_half_half(nvmlClass &nvml);
float gemm_cublas_kernel_float_bf16_float(nvmlClass &nvml);
float gemm_cublas_kernel_float_bf16_bf16(nvmlClass &nvml);
float gemm_cublas_kernel_int_int8_int(nvmlClass &nvml);
