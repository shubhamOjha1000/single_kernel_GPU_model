#include <argparse/argparse.hpp>
#include <gemm/gemm_top.cuh>
#include <gemm/gemm_global.cuh>
#include <util/precision_select.cuh>
#include <iostream>
#include <fstream>

int gdim_M;
int gdim_N;
int gdim_K;
int gnum_iter;
Precision gmulprecision;
Precision gaccprecision;
bool gprint_result;
bool gtensor_cores;
bool gsgemm;
bool gtrn_matA;
bool gtrn_matB;
float gsparsityA;
float gsparsityB;
int gbatch;
bool gstrided_batched;
int device;
bool ghalf_prec_accumulate_full;

int main(int argc, char *argv[])
{
    // Program Title
    std::cout << "[INFO] CUDA Bench - General Matrix-Matrix Multiplication (GEMM) \n";
    std::cout << "[INFO] Version 1.0.0 (C)2022 Bagus Hanindhito \n";
    std::cout << "[INFO] Matrix-Matrix multiplication follows equation: C = (alpha)x(AxB) + (beta)xC\n";
    std::cout << "[INFO] where alpha=1.00, beta=0.00, and A[MxK], B[KxN], and C[MxN] are matrices \n\n\n";

    // Arguments Parser
    argparse::ArgumentParser program(argv[0], "1.0.0", argparse::default_arguments::help);
        program.add_argument("dim_M")
            .help("Positive integer that describes M dimension of the matrices A(MxK) and C(MxN)")
            .scan<'i', int>();
        program.add_argument("dim_N")
            .help("Positive integer that describes N dimension of the matrices B(KxN) and C(MxN)")
            .scan<'i', int>();
        program.add_argument("dim_K")
            .help("Positive integer that describes K dimension of the matrices A(MxK) and B(KxN)")
            .scan<'i', int>();
        program.add_argument("-G", "--gemm")
            .help("Use cublasSgemm (single-precision fp32)")
            .default_value(false)
            .implicit_value(true)
            .metavar("SGEMM");
        program.add_argument("-R", "--result")
            .help("Show result at the end of program")
            .default_value(false)
            .implicit_value(true)
            .metavar("RESULT");
        program.add_argument("-C", "--cudacoresonly")
            .help("Use CUDA Cores only and do not use Tensor Cores")
            .default_value(false)
            .implicit_value(true)
            .metavar("CUDACORES");
        program.add_argument("-M", "--mulprecision")
            .help("Select matrix multiplication precision: fp64, fp32, fp16, int8, or int4")
            .default_value(std::string("fp16"))
            .metavar("MULPREC");
        program.add_argument("-A", "--accprecision")
            .help("Select matrix accumulation precision: fp64, fp32, fp16, int8, or int4")
            .default_value(std::string("fp16"))
            .metavar("ACCPREC");
        program.add_argument("--trn_matA")
            .help("Transpose matrix A?")
            .default_value(false)
            .implicit_value(true)
            .metavar("TRNMATA");
        program.add_argument("--trn_matB")
            .help("Transpose matrix B?")
            .default_value(false)
            .implicit_value(true)
            .metavar("TRNMATB");
        program.add_argument("-I", "--iterations")
            .help("Number of iterations, useful for performance profiling")
            .scan<'i', int>()
            .default_value(1)
            .metavar("ITER");
        program.add_argument("-O", "--nvml_output")
            .help("NVML CSV output save directory/filename")
            .default_value(std::string("../nvml-results/gpuStats.csv"))
            .metavar("NVMLO");
        program.add_argument("-B")
            .help("Batch size for batched matrix multiplication (cublasSgemmBatched)")
            .default_value(1)
            .metavar("BATCH")
            .scan<'i', int>();
        program.add_argument("--strided_batched")
            .help("Whether to use cublasSgemmStridedBatched when the batch size is larger than 1")
            .default_value(false)
            .implicit_value(true)
            .metavar("STRDBATCH");
        program.add_argument("--device")
            .help("Device ID to be used - this program only supports single-GPU run. Should specify the GPU ID (0~#GPU-1) to be used.")
            .default_value(0)
            .metavar("DEVICE")
            .scan<'i', int>();
        program.add_argument("--fp16_accumulate_in_fp32")
            .help("For precM=fp16, precA=fp16, whether the accumulation will happen in fp32")
            .default_value(false)
            .implicit_value(true)
            .metavar("FP16ACCFP32");

    try 
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) 
    {
        std::cerr << "[ERR!] Argument parsing error: " << err.what() << std::endl << std::endl << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    // Argument Processing
    gdim_M = program.get<int>("dim_M");
    gdim_N = program.get<int>("dim_N");
    gdim_K = program.get<int>("dim_K");
    gbatch = program.get<int>("-B");
    gstrided_batched = program.get<bool>("--strided_batched");

    gnum_iter = program.get<int>("--iterations");

    std::string str_mulprecision   = program.get<std::string>("--mulprecision");
    std::string str_accprecision   = program.get<std::string>("--accprecision");

    gprint_result = program.get<bool>("--result");
    gtensor_cores = !(program.get<bool>("--cudacoresonly"));

    std::string filename = program.get<std::string>("--nvml_output");

    gsgemm = program.get<bool>("--gemm");
    gtrn_matA = program.get<bool>("--trn_matA");
    gtrn_matB = program.get<bool>("--trn_matB");

    device = program.get<int>("--device");

    ghalf_prec_accumulate_full = program.get<bool>("--fp16_accumulate_in_fp32");

    // Argument Validation
    if(gdim_M<=0 || gdim_N<=0 || gdim_K<=0)
    {
        std::cerr <<"[ERR!] Argument parsing error: Matrices' dimensions must be positive integers\n\n\n";
        std::cerr << program;
        std::exit(1);
    }
    
    if(gnum_iter<=0)
    {
        std::cerr <<"[ERR!] Argument parsing error: Number of iterations must be positive integers\n\n\n";
        std::cerr << program;
        std::exit(1);
    }

    if      (str_mulprecision=="fp64") {gmulprecision=PRECISION_FP64;}
    else if (str_mulprecision=="fp32") {gmulprecision=PRECISION_FP32;}
    else if (str_mulprecision=="fp16") {gmulprecision=PRECISION_FP16;}
    else if (str_mulprecision=="bf16") {gmulprecision=PRECISION_BF16;}
    else if (str_mulprecision=="int8") {gmulprecision=PRECISION_INT8;}
    else
    {
        std::cerr <<"[ERR!] Argument parsing error: Unsupported matrix multiplication precision\n\n\n";
        std::cerr << program;
        std::exit(1);
    }

    if      (str_accprecision=="fp64") {gaccprecision=PRECISION_FP64;}
    else if (str_accprecision=="fp32") {gaccprecision=PRECISION_FP32;}
    else if (str_accprecision=="fp16") {gaccprecision=PRECISION_FP16;}
    else if (str_accprecision=="bf16") {gaccprecision=PRECISION_BF16;}
    else if (str_accprecision=="int8") {gaccprecision=PRECISION_INT8;}
    else
    {
        std::cerr <<"[ERR!] Argument parsing error: Unsupported matrix accumulation precision\n\n\n";
        std::cerr << program;
        std::exit(1);
    }

    float times = gemm_cublas(filename, device);
 
}
