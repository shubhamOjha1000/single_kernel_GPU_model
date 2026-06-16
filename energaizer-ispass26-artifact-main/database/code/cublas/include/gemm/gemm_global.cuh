// Code from https://github.com/hibagus/CUDA_Bench/blob/main/include/CUDA_Bench/gemm/gemm_global.cuh

// Modifications: remove options that are not used in this code

#pragma once
#include <util/precision_select.cuh>

extern int gdim_M;
extern int gdim_N;
extern int gdim_K;
extern int gnum_iter;
extern Precision gmulprecision;
extern Precision gaccprecision;
extern bool gprint_result;
extern bool gtensor_cores;
extern bool gsgemm;
extern bool gtrn_matA;
extern bool gtrn_matB;
extern int gbatch;
extern bool gstrided_batched;
extern bool ghalf_prec_accumulate_full;