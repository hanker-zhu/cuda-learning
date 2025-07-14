# filepath: pytorch-gemm-test/run_overlap_test.sh
#!/bin/bash

# 使用2个GPU进行测试
nproc_per_node=2
export CUDA_VISIBLE_DEVICES=0,1

# 定义要测试的一系列Grid维度
# 例如，从1个线程块到80个线程块
# A100有108个SMs，所以这个范围可以很好地展示从无重叠到完全重叠的过程
GRID_DIMS_TO_TEST="1 4 8 16 32 40"

torchrun \
    --nproc_per_node=$nproc_per_node \
    --nnodes=1 \
    overlap_perf_test.py \
    --comm-size 4096 \
    --comp-size 4096 \
    --repeats 200 \
    --grid-dims ${GRID_DIMS_TO_TEST}