cmake_minimum_required(VERSION 3.14.3)

set(PROJECT_NAME "CUDA_Bench")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
project(${PROJECT_NAME} C CXX CUDA)

enable_language(CUDA)
find_package(CUDA  REQUIRED)

include_directories("${CUDA_INCLUDE_DIRS}")


set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
set(GPU_ARCHITECTURE_SUPPORT "80")

# CUDA Architecture Support:
# * Fermi (Note: Dropped from CUDA 10 onwards)
#   - 20: GTX 400, GTX 500, GTX 600
# * Kepler (Note: Dropped from CUDA 11 onwards)
#   - 30: GTX 700
#   - 35: Tesla K40
#   - 37: Tesla K80
# * Maxwell
#   - 50: Tesla Mxxxx, Quadro Mxxxx
#   - 52: GTX 900, Titan X, Quadro M6000
#   - 53: TX1, CX, PX, Nano
# * Pascal
#   - 60: Quadro GP100, Tesla P100
#   - 61: GTX 1000, Titan Xp, Tesla P40, Tesla P4
#   - 62: TX2, PX2
# * Volta
#   - 70: Tesla V100, Titan V, Quadro GV100
#   - 72: AGX Xavier, AGX Pegasus, Xavier NX
# * Turing
#   - 75: GTX 1660, RTX 2000, Titan RTX, Quadro RTX xxxx, Quadro Txxxx, Tesla T4 
# * Ampere
#   - 80: A100
#   - 86: RTX 3000, RTX Axxxx, A40, A2, ....
# * Hopper
#   - 90: H100
# * Ada Lovelace
#   - 89: RTX 4000, RTX 6000 (Ada), L40

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")

# interface library for external library
include(cmake/ExternalLibs.cmake)

# add project include dirs for everything
include_directories(${CMAKE_SOURCE_DIR}/include)

# add project library
set(PROJECT_GEMM_LIB "gemmbenchlib" )

# Add Source Code and Apps
add_subdirectory(src)
add_subdirectory(app)
