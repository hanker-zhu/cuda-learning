import torch
import time
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from gemm_test_ext._C import gemm_custom_grid, vector_add

def run_kernel_test(args):
    """运行内核测试并分析SMID分布"""
    # 设置CUDA设备
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print(f"\n使用CUDA设备: {torch.cuda.get_device_name(device)}")
    print(f"设备总SM数量: {torch.cuda.get_device_properties(device).multi_processor_count}")
    
    # 准备GEMM输入
    M, K, N = args.gemm_size, args.gemm_size, args.gemm_size
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    C = torch.zeros(M, N, device=device)
    
    # 准备向量加法输入
    X = torch.randn(args.vec_size, device=device)
    Y = torch.randn(args.vec_size, device=device)
    Z = torch.zeros(args.vec_size, device=device)
    
    # 创建CUDA流
    gemm_stream = torch.cuda.Stream()
    vec_stream = torch.cuda.Stream()
    
    # 测试模式: 串行、并行或都有
    if args.mode in ['serial', 'both']:
        print("\n" + "="*80)
        print("【串行测试模式】: 先运行GEMM，再运行向量加法")
        print("="*80)
        
        # 预热
        print("预热中...")
        for _ in range(args.warmup):
            try:
                gemm_custom_grid(A, B, C, args.gemm_blocks, 1)
                vector_add(X, Y, Z, args.vec_blocks)
            except Exception as e:
                print(f"预热阶段出错: {e}")
                continue
        
        # 计时串行执行
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # 用于存储SMID数据
        gemm_smids = None
        vec_smids = None
        
        try:
            start.record()
            # 先执行GEMM
            gemm_smids = gemm_custom_grid(A, B, C, args.gemm_blocks, 1)
            if gemm_smids is None:
                print("警告: GEMM没有返回SMID数据，检查CUDA扩展实现")
                gemm_smids = torch.zeros(1, device=device, dtype=torch.int32)  # 创建一个虚拟张量
                
            # 再执行向量加法
            vec_smids = vector_add(X, Y, Z, args.vec_blocks)
            if vec_smids is None:
                print("警告: 向量加法没有返回SMID数据，检查CUDA扩展实现")
                vec_smids = torch.zeros(1, device=device, dtype=torch.int32)  # 创建一个虚拟张量
                
            end.record()
            
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)
            print(f"串行执行总时间: {elapsed_time:.4f} ms")
            
            # 分析串行执行的SMID
            if gemm_smids is not None and vec_smids is not None:
                analyze_smids(gemm_smids, vec_smids, "serial")
            else:
                print("无法分析SMID: 数据无效")
                
        except Exception as e:
            print(f"串行执行过程中出错: {e}")
            import traceback
            traceback.print_exc()
    
    if args.mode in ['parallel', 'both']:
        print("\n" + "="*80)
        print("【并行测试模式】: 在不同流中同时运行GEMM和向量加法")
        print("="*80)
        
        # 预热
        print("预热中...")
        for _ in range(args.warmup):
            try:
                with torch.cuda.stream(gemm_stream):
                    gemm_custom_grid(A, B, C, args.gemm_blocks, 1)
                with torch.cuda.stream(vec_stream):
                    vector_add(X, Y, Z, args.vec_blocks)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"预热阶段出错: {e}")
                continue
        
        # 计时并行执行
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # 用于存储SMID数据
        parallel_gemm_smids = None
        parallel_vec_smids = None
        
        try:
            start.record()
            # 并行执行
            with torch.cuda.stream(gemm_stream):
                parallel_gemm_smids = gemm_custom_grid(A, B, C, args.gemm_blocks, 1)
            with torch.cuda.stream(vec_stream):
                parallel_vec_smids = vector_add(X, Y, Z, args.vec_blocks)
            torch.cuda.synchronize()
            end.record()
            
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)
            print(f"并行执行总时间: {elapsed_time:.4f} ms")
            
            # 检查返回的SMID数据是否有效
            if parallel_gemm_smids is None:
                print("警告: GEMM没有返回并行SMID数据，检查CUDA扩展实现")
                parallel_gemm_smids = torch.zeros(1, device=device, dtype=torch.int32)
                
            if parallel_vec_smids is None:
                print("警告: 向量加法没有返回并行SMID数据，检查CUDA扩展实现")
                parallel_vec_smids = torch.zeros(1, device=device, dtype=torch.int32)
                
            # 分析并行执行的SMID
            if parallel_gemm_smids is not None and parallel_vec_smids is not None:
                analyze_smids(parallel_gemm_smids, parallel_vec_smids, "parallel")
            else:
                print("无法分析并行SMID: 数据无效")
                
        except Exception as e:
            print(f"并行执行过程中出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 如果同时执行了串行和并行测试，显示比较结果
    if args.mode == 'both' and 'elapsed_time_serial' in locals() and 'elapsed_time_parallel' in locals():
        speedup = elapsed_time_serial / elapsed_time_parallel if elapsed_time_parallel > 0 else float('inf')
        print(f"\n并行执行比串行执行快: {speedup:.2f}x")

def analyze_smids(gemm_smids, vec_smids, test_mode):
    """分析并可视化SMID分布"""
    # 添加数据检查
    if gemm_smids is None or vec_smids is None:
        print(f"错误: {test_mode}模式下SMID数据无效")
        return None, None, None
    
    try:
        # 转换为numpy数组
        gemm_smids_np = gemm_smids.cpu().numpy().flatten() if torch.is_tensor(gemm_smids) else np.array([])
        vec_smids_np = vec_smids.cpu().numpy().flatten() if torch.is_tensor(vec_smids) else np.array([])
        
        # 检查数据是否为空
        if len(gemm_smids_np) == 0 or len(vec_smids_np) == 0:
            print(f"警告: {test_mode}模式下SMID数据为空")
            return np.array([]), np.array([]), np.array([])
        
        # 获取唯一的SMID及其计数
        gemm_unique, gemm_counts = np.unique(gemm_smids_np, return_counts=True)
        vec_unique, vec_counts = np.unique(vec_smids_np, return_counts=True)
        
        # 分析重叠
        overlap_smids = np.intersect1d(gemm_unique, vec_unique)
        
        # 打印分析结果
        print(f"\n【SMID分布分析 - {test_mode}】")
        print(f"GEMM使用的SM数量: {len(gemm_unique)}, IDs: {gemm_unique}")
        print(f"向量加法使用的SM数量: {len(vec_unique)}, IDs: {vec_unique}")
        print(f"重叠使用的SM数量: {len(overlap_smids)}")
        
        if len(overlap_smids) > 0:
            print(f"重叠的SM IDs: {overlap_smids}")
        
        # 创建SMID分布图
        create_smid_distribution_plot(gemm_unique, vec_unique, test_mode)
        
        return gemm_unique, vec_unique, overlap_smids
    except Exception as e:
        print(f"SMID分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return np.array([]), np.array([]), np.array([])

def create_smid_distribution_plot(gemm_smids, vec_smids, test_mode):
    """创建SMID分布可视化图"""
    # 获取所有唯一的SMID
    all_smids = np.union1d(gemm_smids, vec_smids)
    all_smids.sort()
    
    # 创建数据集
    gemm_data = np.isin(all_smids, gemm_smids).astype(int)
    vec_data = np.isin(all_smids, vec_smids).astype(int) * 2
    
    # 创建叠加图
    overlap_data = gemm_data + vec_data
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 创建颜色映射
    cmap = plt.cm.get_cmap('viridis', 4)
    
    # 绘制热图
    plt.imshow([overlap_data], aspect='auto', cmap=cmap)
    plt.colorbar(ticks=[0, 1, 2, 3], 
                 label='SM使用情况',
                 orientation='vertical')
    
    # 添加标签
    plt.yticks([])
    plt.xticks(range(len(all_smids)), all_smids)
    plt.xlabel('SM ID')
    plt.title(f'GEMM和向量加法使用的SM分布 ({test_mode})')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap(0), label='未使用'),
        Patch(facecolor=cmap(1), label='仅GEMM'),
        Patch(facecolor=cmap(2), label='仅向量加法'),
        Patch(facecolor=cmap(3), label='重叠使用')
    ]
    plt.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    # 保存图表
    output_file = f'smid_distribution_{test_mode}.png'
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"SMID分布图已保存为 '{output_file}'")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUDA内核SM分配测试工具")
    
    # 测试模式子命令
    subparsers = parser.add_subparsers(dest='command', help='测试命令')
    
    # 单次测试
    single_parser = subparsers.add_parser('single', help='运行单次测试')
    single_parser.add_argument('--gemm-size', type=int, default=4096, help="GEMM矩阵的维度")
    single_parser.add_argument('--vec-size', type=int, default=1024*1024*16, help="向量大小")
    single_parser.add_argument('--gemm-blocks', type=int, required=True, help="GEMM线程块数量")
    single_parser.add_argument('--vec-blocks', type=int, required=True, help="向量加法线程块数量")
    single_parser.add_argument('--mode', choices=['serial', 'parallel', 'both'], default='both', 
                              help="测试模式: serial(串行), parallel(并行), both(两者)")
    single_parser.add_argument('--warmup', type=int, default=5, help="预热运行次数")
    
    # 扫描测试
    sweep_parser = subparsers.add_parser('sweep', help='运行SM分配扫描测试')
    sweep_parser.add_argument('--gemm-size', type=int, default=4096, help="GEMM矩阵的维度")
    sweep_parser.add_argument('--vec-size', type=int, default=1024*1024*16, help="向量大小")
    sweep_parser.add_argument('--warmup', type=int, default=3, help="每次测试的预热运行次数")
    
    args = parser.parse_args()
    
    try:
        if args.command == 'single':
            run_kernel_test(args)
        elif args.command == 'sweep':
            run_sweep_test(args)
        else:
            parser.print_help()
    except Exception as e:
        print(f"程序执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
