import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from gemm_test_ext._C import gemm_custom_grid, vector_add

def collect_smid_data(gemm_size, vec_size, gemm_blocks, vec_blocks):
    """收集GEMM和向量加法操作的SMID数据"""
    device = 'cuda:0'
    torch.cuda.set_device(device)
    
    # 准备GEMM输入
    M, K, N = gemm_size, gemm_size, gemm_size
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    C = torch.zeros(M, N, device=device)
    
    # 准备向量加法输入
    X = torch.randn(vec_size, device=device)
    Y = torch.randn(vec_size, device=device)
    Z = torch.zeros(vec_size, device=device)
    
    # 串行执行并收集SMID
    print("正在收集串行执行的SMID数据...")
    gemm_smids = gemm_custom_grid(A, B, C, gemm_blocks, 1)
    vecadd_smids = vector_add(X, Y, Z, vec_blocks)
    
    # 并行执行并收集SMID
    print("正在收集并行执行的SMID数据...")
    gemm_stream = torch.cuda.Stream()
    vecadd_stream = torch.cuda.Stream()
    
    with torch.cuda.stream(gemm_stream):
        parallel_gemm_smids = gemm_custom_grid(A, B, C, gemm_blocks, 1)
    with torch.cuda.stream(vecadd_stream):
        parallel_vecadd_smids = vector_add(X, Y, Z, vec_blocks)
    torch.cuda.synchronize()
    
    return {
        'serial': {
            'gemm': gemm_smids,
            'vecadd': vecadd_smids
        },
        'parallel': {
            'gemm': parallel_gemm_smids,
            'vecadd': parallel_vecadd_smids
        }
    }

def analyze_smid_overlap(smid_data):
    """分析SMID重叠情况"""
    # 处理串行执行的SMID
    serial_gemm_smids = smid_data['serial']['gemm'].cpu().numpy().flatten()
    serial_vecadd_smids = smid_data['serial']['vecadd'].cpu().numpy().flatten()
    
    # 处理并行执行的SMID
    parallel_gemm_smids = smid_data['parallel']['gemm'].cpu().numpy().flatten()
    parallel_vecadd_smids = smid_data['parallel']['vecadd'].cpu().numpy().flatten()
    
    # 获取唯一SMID
    serial_gemm_unique = np.unique(serial_gemm_smids)
    serial_vecadd_unique = np.unique(serial_vecadd_smids)
    parallel_gemm_unique = np.unique(parallel_gemm_smids)
    parallel_vecadd_unique = np.unique(parallel_vecadd_smids)
    
    # 计算重叠
    serial_overlap = np.intersect1d(serial_gemm_unique, serial_vecadd_unique)
    parallel_overlap = np.intersect1d(parallel_gemm_unique, parallel_vecadd_unique)
    
    # 打印结果
    print("\n--- SMID分布分析 ---")
    print("串行执行:")
    print(f"  GEMM使用的SM数量: {len(serial_gemm_unique)}, SM IDs: {serial_gemm_unique}")
    print(f"  向量加法使用的SM数量: {len(serial_vecadd_unique)}, SM IDs: {serial_vecadd_unique}")
    print(f"  重叠使用的SM数量: {len(serial_overlap)}, SM IDs: {serial_overlap}")
    
    print("\n并行执行:")
    print(f"  GEMM使用的SM数量: {len(parallel_gemm_unique)}, SM IDs: {parallel_gemm_unique}")
    print(f"  向量加法使用的SM数量: {len(parallel_vecadd_unique)}, SM IDs: {parallel_vecadd_unique}")
    print(f"  重叠使用的SM数量: {len(parallel_overlap)}, SM IDs: {parallel_overlap}")
    
    # 创建详细的分布图
    visualize_detailed_distribution(
        serial_gemm_unique, serial_vecadd_unique,
        parallel_gemm_unique, parallel_vecadd_unique
    )
    
    return {
        'serial_overlap': serial_overlap,
        'parallel_overlap': parallel_overlap
    }

def visualize_detailed_distribution(serial_gemm, serial_vecadd, parallel_gemm, parallel_vecadd):
    """创建详细的SM分布可视化"""
    # 获取所有唯一的SMID
    all_smids = np.unique(np.concatenate([
        serial_gemm, serial_vecadd, parallel_gemm, parallel_vecadd
    ]))
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 在串行执行图上绘制
    serial_data = np.zeros((2, len(all_smids)))
    for i, smid in enumerate(all_smids):
        if smid in serial_gemm:
            serial_data[0, i] = 1
        if smid in serial_vecadd:
            serial_data[1, i] = 1
    
    # 在并行执行图上绘制
    parallel_data = np.zeros((2, len(all_smids)))
    for i, smid in enumerate(all_smids):
        if smid in parallel_gemm:
            parallel_data[0, i] = 1
        if smid in parallel_vecadd:
            parallel_data[1, i] = 1
    
    # 绘制热图
    im1 = axes[0].imshow(serial_data, cmap='Blues', aspect='auto')
    im2 = axes[1].imshow(parallel_data, cmap='Reds', aspect='auto')
    
    # 添加标签
    axes[0].set_title('串行执行SM分布')
    axes[1].set_title('并行执行SM分布')
    
    for ax in axes:
        ax.set_xticks(range(len(all_smids)))
        ax.set_xticklabels(all_smids)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['GEMM', '向量加法'])
        ax.set_xlabel('SM ID')
    
    # 添加颜色条
    fig.colorbar(im1, ax=axes[0], label='使用状态')
    fig.colorbar(im2, ax=axes[1], label='使用状态')
    
    plt.tight_layout()
    plt.savefig('detailed_smid_distribution.png')
    print("详细的SMID分布图已保存为 'detailed_smid_distribution.png'")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析CUDA核函数的SMID分布")
    parser.add_argument('--gemm-size', type=int, default=4096, help="GEMM矩阵的维度")
    parser.add_argument('--vec-size', type=int, default=1024*1024*16, help="向量大小")
    parser.add_argument('--gemm-blocks', type=int, default=40, help="GEMM线程块数量")
    parser.add_argument('--vec-blocks', type=int, default=40, help="向量加法线程块数量")
    args = parser.parse_args()
    
    smid_data = collect_smid_data(
        args.gemm_size, args.vec_size, 
        args.gemm_blocks, args.vec_blocks
    )
    
    overlap_data = analyze_smid_overlap(smid_data)
