import torch
import time
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from gemm_test_ext._C import gemm_custom_grid, vector_add

def analyze_smid_distribution(gemm_smids, vecadd_smids):
    """分析GEMM和向量加法使用的SM分布情况"""
    # 检查输入是否为None
    if gemm_smids is None or vecadd_smids is None:
        print("警告: 无法分析SMID分布 - 输入数据为None")
        return None, None, None
    
    # 将张量转换为numpy数组以便分析
    if torch.is_tensor(gemm_smids):
        gemm_smids = gemm_smids.cpu().numpy().flatten()
    if torch.is_tensor(vecadd_smids):
        vecadd_smids = vecadd_smids.cpu().numpy().flatten()
    
    # 过滤掉无效值（内核中+1，所以这里-1后应>=0）
    gemm_smids = gemm_smids - 1
    vecadd_smids = vecadd_smids - 1
    gemm_smids = gemm_smids[gemm_smids >= 0]
    vecadd_smids = vecadd_smids[vecadd_smids >= 0]
    
    # 检查过滤后是否还有数据
    if len(gemm_smids) == 0 and len(vecadd_smids) == 0:
        print("警告: 过滤无效值后没有SMID数据")
        return None, None, None
    
    # 获取唯一的SMID值及其计数
    gemm_unique_smids, gemm_counts = np.unique(gemm_smids, return_counts=True)
    vecadd_unique_smids, vecadd_counts = np.unique(vecadd_smids, return_counts=True)
    
    print(f"GEMM使用的SM数量: {len(gemm_unique_smids)} -> {dict(zip(gemm_unique_smids, gemm_counts))}")
    print(f"向量加法使用的SM数量: {len(vecadd_unique_smids)} -> {dict(zip(vecadd_unique_smids, vecadd_counts))}")
    
    # 计算重叠的SM
    overlap_smids = np.intersect1d(gemm_unique_smids, vecadd_unique_smids)
    print(f"重叠使用的SM数量: {len(overlap_smids)}")
    if len(overlap_smids) > 0:
        print(f"重叠的SM IDs: {overlap_smids}")
    
    # 可视化SMID分布
    visualize_smid_distribution(gemm_unique_smids, vecadd_unique_smids, overlap_smids)
    
    return gemm_unique_smids, vecadd_unique_smids, overlap_smids

def visualize_smid_distribution(gemm_smids, vecadd_smids, overlap_smids):
    """可视化GEMM和向量加法使用的SM分布"""
    if gemm_smids is None or vecadd_smids is None or overlap_smids is None:
        print("无法可视化SMID分布，数据不完整。")
        return
        
    # 确定所有唯一的SMID值
    all_smids = np.union1d(gemm_smids, vecadd_smids)
    if len(all_smids) == 0:
        print("没有有效的SMID可供可视化。")
        return
    
    # 创建数据集
    gemm_data = np.isin(all_smids, gemm_smids).astype(int)
    vecadd_data = np.isin(all_smids, vecadd_smids).astype(int) * 2
    
    # 创建叠加图，重叠区域会显示为3
    data = gemm_data + vecadd_data
    
    # 创建图表
    plt.figure(figsize=(max(12, len(all_smids) // 4), 6))
    
    # 创建颜色映射
    cmap = plt.cm.get_cmap('viridis', 4)
    
    # 绘制热图
    plt.imshow([data], aspect='auto', cmap=cmap)
    plt.colorbar(ticks=[0.375, 1.125, 1.875, 2.625], 
                 label='SM usage',
                 orientation='vertical')
    plt.gca().get_yticklabels()[0].set_text('not use')
    plt.gca().get_yticklabels()[1].set_text('only GEMM')
    plt.gca().get_yticklabels()[2].set_text('only VecAdd')
    plt.gca().get_yticklabels()[3].set_text('overlap use')
    
    # 添加标签
    plt.yticks([])
    plt.xticks(range(len(all_smids)), all_smids)
    plt.xlabel('SM ID')
    plt.title('GEMM和向量加法使用的SM分布')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap(0/3.), label='not use'),
        Patch(facecolor=cmap(1/3.), label='only GEMM'),
        Patch(facecolor=cmap(2/3.), label='only VecAdd'),
        Patch(facecolor=cmap(3/3.), label='overlap use')
    ]
    plt.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, -0.2), ncol=4)
    
    # 保存图表
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('smid_distribution.png', bbox_inches='tight')
    print(f"SMID分布图已保存为 'smid_distribution.png'")
    
    # 关闭图表
    plt.close()

def measure_performance(A, B, C, X, Y, Z, gemm_blocks, vec_add_blocks, repeats, warmup=10, sleep_ms=0.0):
    """
    测量串行和并行执行的性能，并进行比较。
    - gemm_blocks: 分配给GEMM核函数的线程块数量 (近似于SM数量)
    - vec_add_blocks: 分配给向量加法核函数的线程块数量 (近似于SM数量)
    - sleep_ms: 可选的人工延迟，以毫秒为单位
    """
    # 如果其中一个任务的线程块为0，则无法执行
    if gemm_blocks <= 0 or vec_add_blocks <= 0:
        return float('nan'), float('nan'), (None, None), (None, None)
    
    # 如果重复次数为0，则无法测量
    if repeats <= 0:
        return float('nan'), float('nan'), (None, None), (None, None)

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
    torch.cuda.synchronize() # 确保数据已生成
    
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

    # 在计时结束后，再运行一次以捕获SMID
    gemm_smids = gemm_custom_grid(A_inputs[-1], B_inputs[-1], C_outputs[-1], gemm_blocks, 1)
    vecadd_smids = vector_add(X_inputs[-1], Y_inputs[-1], Z_outputs[-1], vec_add_blocks)
    torch.cuda.synchronize()

    # --- 3. 测量并行总时间 ---
    overlap_start_event = torch.cuda.Event(enable_timing=True)
    overlap_end_event = torch.cuda.Event(enable_timing=True)
    
    overlap_start_event.record()
    for i in range(repeats):
        with torch.cuda.stream(gemm_stream):
            gemm_custom_grid(A_inputs[i], B_inputs[i], C_outputs[i], gemm_blocks, 1)
        
        with torch.cuda.stream(vec_add_stream):
            vector_add(X_inputs[i], Y_inputs[i], Z_outputs[i], vec_add_blocks)
    
    # 两个流都需要同步到主事件
    gemm_stream.record_event(overlap_end_event)
    vec_add_stream.record_event(overlap_end_event)
    
    torch.cuda.synchronize()
    overlap_total_time = overlap_start_event.elapsed_time(overlap_end_event) / repeats

    # 在计时结束后，再运行一次以捕获SMID
    with torch.cuda.stream(gemm_stream):
        parallel_gemm_smids = gemm_custom_grid(A_inputs[-1], B_inputs[-1], C_outputs[-1], gemm_blocks, 1)
    with torch.cuda.stream(vec_add_stream):
        parallel_vecadd_smids = vector_add(X_inputs[-1], Y_inputs[-1], Z_outputs[-1], vec_add_blocks)
    torch.cuda.synchronize()

    return serial_total_time, overlap_total_time, (gemm_smids, vecadd_smids), (parallel_gemm_smids, parallel_vecadd_smids)


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

    serial_time, overlap_time, serial_smids, parallel_smids = measure_performance(
        A, B, C, X, Y, Z, args.gemm_sms, args.vec_add_sms, args.repeats, args.warmup, args.sleep_ms
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
    
    # 分析串行执行的SMID分布
    print("\n--- 串行执行SMID分析 ---")
    if serial_smids is not None and all(s is not None for s in serial_smids):
        gemm_smids, vecadd_smids = serial_smids
        analyze_smid_distribution(gemm_smids, vecadd_smids)
    else:
        print("警告: 无法获取串行执行的SMID数据")
    
    # 分析并行执行的SMID分布
    print("\n--- 并行执行SMID分析 ---")
    if parallel_smids is not None and all(s is not None for s in parallel_smids):
        parallel_gemm_smids, parallel_vecadd_smids = parallel_smids
        analyze_smid_distribution(parallel_gemm_smids, parallel_vecadd_smids)
    else:
        print("警告: 无法获取并行执行的SMID数据")
    
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
    # 新增参数：人工延迟
    parser.add_argument('--sleep-ms', type=float, default=0.0, help="内核执行中添加的人工延迟（毫秒）")
    args = parser.parse_args()
    
    results = run_experiment(args)