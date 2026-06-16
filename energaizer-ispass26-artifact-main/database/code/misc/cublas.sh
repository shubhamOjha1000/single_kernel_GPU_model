#!/bin/bash

# 1. Check git modules
git submodule update --init --recursive

# 2. Build cublas benchmarking setup
cd cublas
mkdir build
mkdir bin
cd build
cmake ..
make -j20
cd ../..