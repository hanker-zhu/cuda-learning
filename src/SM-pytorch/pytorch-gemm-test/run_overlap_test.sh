#!/bin/bash

# 使用单个GPU进行测试
export CUDA_VISIBLE_DEVICES=0

# 编译扩展
echo "===== 正在编译CUDA扩展... ====="
python3 setup.py install --force
if [ $? -ne 0 ]; then
    echo "编译失败，请检查错误信息。"
    exit 1
fi
echo "编译完成。"

# 运行测试
echo -e "\n===== 运行CUDA内核SM分配测试 ====="

# 定义测试参数
GEMM_SMS=40
VEC_ADD_SMS=40
GEMM_SIZE=4096
VEC_SIZE=$((1024*1024*16))
REPEATS=20
WARMUP=10

echo "测试配置:"
echo "  - 分配给GEMM的SMs: $GEMM_SMS"
echo "  - 分配给向量加法的SMs: $VEC_ADD_SMS"
echo "  - GEMM矩阵大小: ${GEMM_SIZE}x${GEMM_SIZE}"
echo "  - 向量大小: $VEC_SIZE"
echo "  - 重复次数: $REPEATS"
echo "  - 预热次数: $WARMUP"

# 执行Python脚本
python3 overlap_perf_test.py \
    --gemm-sms $GEMM_SMS \
    --vec-add-sms $VEC_ADD_SMS \
    --gemm-size $GEMM_SIZE \
    --vec-size $VEC_SIZE \
    --repeats $REPEATS \
    --warmup $WARMUP

if [ $? -eq 0 ]; then
    echo -e "\n测试完成！性能结果已打印，SMID分布图已保存为 'smid_distribution.png'。"
else
    echo -e "\n测试运行失败。"
fi
