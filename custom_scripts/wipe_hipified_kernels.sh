#!/bin/bash


#Remove ninja build and hipified files from fused kernels
rm -rf megatron/fused_kernels/build
rm -rf megatron/fused_kernels/*hip*