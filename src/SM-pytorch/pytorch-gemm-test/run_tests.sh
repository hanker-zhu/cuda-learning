#!/bin/bash

# 确保CUDA设备可见
export CUDA_VISIBLE_DEVICES=0

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}===== CUDA内核SM分配测试 =====${NC}"

# 确保模块已经安装
pip install -e . > /dev/null 2>&1

# 参数解析
TEST_TYPE=$1  # 单次测试(single)或扫描(sweep)
GEMM_SMS=$2   # GEMM的SM数量
VECADD_SMS=$3 # 向量加法的SM数量
MODE=$4       # 运行模式: gemm, vecadd, both

if [ "$TEST_TYPE" == "single" ]; then
    # 单次测试，使用指定的SM分配
    echo "运行单次测试: GEMM SMs=$GEMM_SMS, VecAdd SMs=$VECADD_SMS, 模式=$MODE"
    
    # 根据模式运行不同的测试
    if [ "$MODE" == "gemm" ] || [ "$MODE" == "both" ]; then
        echo "  运行GEMM测试..."
        python overlap_perf_test.py --gemm-sms $GEMM_SMS --vec-add-sms 0
    fi
    
    if [ "$MODE" == "vecadd" ] || [ "$MODE" == "both" ]; then
        echo "  运行向量加法测试..."
        python overlap_perf_test.py --gemm-sms 0 --vec-add-sms $VECADD_SMS
    fi
    
    if [ "$MODE" == "both" ]; then
        echo "  运行并发测试..."
        python overlap_perf_test.py --gemm-sms $GEMM_SMS --vec-add-sms $VECADD_SMS
    fi
    
elif [ "$TEST_TYPE" == "sweep" ]; then
    # 扫描测试，尝试不同的SM分配组合
    echo "运行SM分配扫描测试..."
    
    # 获取GPU的总SM数量
    SM_COUNT=$(nvidia-smi --query-gpu=multiprocessorCount --format=csv,noheader,nounits)
    echo "检测到GPU的SM数量: $SM_COUNT"
    
    # 创建临时脚本运行扫描测试
    cat > sweep_test.py << EOF
import torch
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import re
import os

# 扫描不同的SM分配组合
sm_count = $SM_COUNT
results = []

# 尝试不同的分配比例
ratios = np.linspace(0.1, 0.9, 9)
for ratio in ratios:
    gemm_sms = int(sm_count * ratio)
    vecadd_sms = sm_count - gemm_sms
    
    # 跳过无效的分配
    if gemm_sms == 0 or vecadd_sms == 0:
        continue
    
    # 运行测试
    print(f"测试 GEMM SMs={gemm_sms}, VecAdd SMs={vecadd_sms}")
    cmd = f"python overlap_perf_test.py --gemm-sms {gemm_sms} --vec-add-sms {vecadd_sms}"
    output = subprocess.check_output(cmd, shell=True, text=True)
    
    # 从输出中提取结果
    serial_time = re.search(r"Serial Time \(ms\).*?(\d+\.\d+)", output)
    overlap_time = re.search(r"Overlap Time \(ms\).*?(\d+\.\d+)", output)
    
    if serial_time and overlap_time:
        serial_time = float(serial_time.group(1))
        overlap_time = float(overlap_time.group(1))
        perf_gain = (serial_time - overlap_time) / serial_time if serial_time > 0 else 0
        
        results.append({
            'gemm_ratio': ratio,
            'gemm_sms': gemm_sms,
            'vecadd_sms': vecadd_sms,
            'serial_time': serial_time,
            'overlap_time': overlap_time,
            'perf_gain': perf_gain
        })
        print(f"  性能增益: {perf_gain:.2%}")

# 绘制结果
if results:
    plt.figure(figsize=(12, 8))
    
    # 提取数据
    ratios = [r['gemm_ratio'] for r in results]
    gains = [r['perf_gain'] for r in results]
    
    # 绘制性能增益与SM分配比例的关系
    plt.subplot(2, 1, 1)
    plt.plot(ratios, gains, 'o-', linewidth=2)
    plt.xlabel('GEMM分配SM比例')
    plt.ylabel('性能增益')
    plt.title('SM分配对并发执行性能增益的影响')
    plt.grid(True)
    
    # 绘制运行时间
    plt.subplot(2, 1, 2)
    serial_times = [r['serial_time'] for r in results]
    overlap_times = [r['overlap_time'] for r in results]
    
    bar_width = 0.35
    x = np.arange(len(ratios))
    plt.bar(x - bar_width/2, serial_times, bar_width, label='串行执行')
    plt.bar(x + bar_width/2, overlap_times, bar_width, label='并行执行')
    
    plt.xlabel('GEMM分配SM比例')
    plt.ylabel('运行时间 (ms)')
    plt.title('SM分配对运行时间的影响')
    plt.xticks(x, [f"{r:.2f}" for r in ratios])
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sm_allocation_sweep.png')
    print("扫描结果已保存为 'sm_allocation_sweep.png'")
EOF
    
    # 运行扫描测试
    python sweep_test.py
    
    # 清理临时脚本
    rm sweep_test.py
    
else
    echo "未知的测试类型: $TEST_TYPE"
    echo "用法: $0 {single|sweep} [GEMM_SMS] [VECADD_SMS] [gemm|vecadd|both]"
    exit 1
fi

echo "测试完成!"
