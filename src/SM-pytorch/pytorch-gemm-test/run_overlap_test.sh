#!/bin/bash

# 使用单个GPU进行测试
export CUDA_VISIBLE_DEVICES=0

# 针对 Tesla T4 (40 SMs) 优化测试参数
# 测试将覆盖资源占用不足、饱和以及超额订阅的情况

# for gemm_sms in 1~40, 
for gemm_sms in 40
do
    # 为不同场景设置vec_add_sms值
    # 分别测试不同资源占用情况
    vec_add_sms=40
    
    echo "--- Testing with GEMM SMs: ${gemm_sms}, VecAdd SMs: ${vec_add_sms} ---"
    python3 overlap_perf_test.py \
        --gemm-size 4096 \
        --vec-size 16777216 \
        --repeats 20 \
        --warmup 5 \
        --gemm-sms ${gemm_sms} \
        --vec-add-sms ${vec_add_sms}
done

#统一调整得更大一点
