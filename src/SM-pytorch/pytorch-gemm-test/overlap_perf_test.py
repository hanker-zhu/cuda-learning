import torch
import time
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from gemm_test_ext._C import gemm_custom_grid, vector_add

def measure_performance(A, B, C, X, Y, Z, gemm_blocks, vec_add_blocks, repeats, warmup=10):
    """
    测量串行和并行执行的性能，并进行比较。
    - gemm_blocks: 分配给GEMM核函数的线程块数量 (近似于SM数量)
    - vec_add_blocks: 分配给向量加法核函数的线程块数量 (近似于SM数量)
    """
    # 如果其中一个任务的线程块为0，则无法执行
    if gemm_blocks <= 0 or vec_add_blocks <= 0:
        return float('nan'), float('nan')

    gemm_stream = torch.cuda.Stream()
    vec_add_stream = torch.cuda.Stream()
    
    # 获取张量形状以便生成新数据
    M, K = A.shape
    K, N = B.shape
    vec_size = X.shape[0]
    device = A.device
    
    # 为每次重复生成不同的输入数据
    A_inputs = [torch.randn(M, K, device=device) for _ in range(repeats)]
    B_inputs = [torch.randn(K, N, device=device) for _ in range(repeats)]
    C_outputs = [torch.zeros(M, N, device=device) for _ in range(repeats)]
    
    X_inputs = [torch.randn(vec_size, device=device) for _ in range(repeats)]
    Y_inputs = [torch.randn(vec_size, device=device) for _ in range(repeats)]
    Z_outputs = [torch.zeros(vec_size, device=device) for _ in range(repeats)]
    
    # --- 1. 预热阶段 ---
    for _ in range(warmup):
        with torch.cuda.stream(gemm_stream):
            # GEMM的grid是二维的，我们假设它在两个维度上均匀分布
            gemm_custom_grid(A, B, C, gemm_blocks, 1)
        with torch.cuda.stream(vec_add_stream):
            vector_add(X, Y, Z, vec_add_blocks)
        torch.cuda.synchronize()

    # --- 2. 测量串行总时间 ---
    serial_start_event = torch.cuda.Event(enable_timing=True)
    serial_end_event = torch.cuda.Event(enable_timing=True)
    
    serial_start_event.record()
    # 先执行所有GEMM，每次使用不同输入
    for i in range(repeats):
        gemm_custom_grid(A_inputs[i], B_inputs[i], C_outputs[i], gemm_blocks, 1)
    # 再执行所有向量加法，每次使用不同输入
    for i in range(repeats):
        vector_add(X_inputs[i], Y_inputs[i], Z_outputs[i], vec_add_blocks)
    serial_end_event.record()
    
    torch.cuda.synchronize()
    serial_total_time = serial_start_event.elapsed_time(serial_end_event) / repeats

    # --- 3. 测量并行总时间 ---
    overlap_start_event = torch.cuda.Event(enable_timing=True)
    overlap_end_event = torch.cuda.Event(enable_timing=True)

    overlap_start_event.record()
    for i in range(repeats):
        with torch.cuda.stream(gemm_stream):
            gemm_custom_grid(A_inputs[i], B_inputs[i], C_outputs[i], gemm_blocks, 1)
        with torch.cuda.stream(vec_add_stream):
            vector_add(X_inputs[i], Y_inputs[i], Z_outputs[i], vec_add_blocks)
        torch.cuda.synchronize()
        
    overlap_end_event.record()
    
    torch.cuda.synchronize()
    overlap_total_time = overlap_start_event.elapsed_time(overlap_end_event) / repeats

    return serial_total_time, overlap_total_time


def run_experiment(args):
    """主实验函数"""
    device = 'cuda:0'
    torch.cuda.set_device(device)
    
    M, K, N = args.gemm_size, args.gemm_size, args.gemm_size
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    C = torch.zeros(M, N, device=device)

    X = torch.randn(args.vec_size, device=device)
    Y = torch.randn(args.vec_size, device=device)
    Z = torch.zeros(args.vec_size, device=device)

    print("\n" + "="*100)
    header = (f"{'GEMM SMs':<12} | {'VecAdd SMs':<12} | {'Serial Time (ms)':<20} | "
              f"{'Overlap Time (ms)':<20} | {'Performance Gain':<20}")
    print(header)
    print("-" * 100)

    # 收集结果用于绘图
    results = []

    serial_time, overlap_time = measure_performance(
        A, B, C, X, Y, Z, args.gemm_sms, args.vec_add_sms, args.repeats, args.warmup
    )
    
    if not math.isnan(serial_time):
        perf_gain = (serial_time - overlap_time) / serial_time if serial_time > 0 else 0.0
        print(f"{args.gemm_sms:<12} | {args.vec_add_sms:<12} | {serial_time:<20.4f} | "
              f"{overlap_time:<20.4f} | {f'{perf_gain:.2%}':<20}")
        
        # 保存结果
        results.append({
            'gemm_sms': args.gemm_sms,
            'vec_add_sms': args.vec_add_sms,
            'serial_time': serial_time,
            'overlap_time': overlap_time,
            'perf_gain': perf_gain
        })
    
    # 返回收集到的结果
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试两个计算核函数并发执行时的性能干扰")
    parser.add_argument('--gemm-size', type=int, default=4096, help="GEMM矩阵的维度 (M, K, N)")
    parser.add_argument('--vec-size', type=int, default=1024*1024*16, help="向量加法的向量大小")
    parser.add_argument('--repeats', type=int, default=20, help="每次测量的重复次数")
    parser.add_argument('--warmup', type=int, default=5, help="预热运行的次数")
    # 新增参数：直接指定分配给每个kernel的SM（线程块）数量
    parser.add_argument('--gemm-sms', type=int, required=True, help="分配给GEMM核函数的SM(线程块)数量")
    parser.add_argument('--vec-add-sms', type=int, required=True, help="分配给向量加法核函数的SM(线程块)数量")
    args = parser.parse_args()
    
    results = run_experiment(args)