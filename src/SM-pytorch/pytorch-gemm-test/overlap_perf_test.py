import torch
import torch.distributed as dist
import os
import time
import argparse
from gemm_test_ext._C import gemm_custom_grid

# 函数签名已更新，接收独立的参数
def timed_comm(input_tensor, output_tensor):
    """单独测量通信操作的时间"""
    dist.barrier()
    # 预热
    for _ in range(5):
        dist.all_to_all_single(output_tensor, input_tensor)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(args.repeats):
        dist.all_to_all_single(output_tensor, input_tensor)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    dist.barrier()
    return (end_time - start_time) / args.repeats

# 函数签名已更新
def timed_comp(A, B, C, grid_dim_x, grid_dim_y):
    """单独测量计算操作的时间"""
    dist.barrier()
    # 预热
    for _ in range(5):
        gemm_custom_grid(A, B, C, grid_dim_x, grid_dim_y)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(args.repeats):
        gemm_custom_grid(A, B, C, grid_dim_x, grid_dim_y)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    dist.barrier()
    return (end_time - start_time) / args.repeats

# 函数签名已更新
def timed_overlap(input_tensor, output_tensor, A, B, C, grid_dim_x, grid_dim_y):
    """测量计算和通信重叠时的时间"""
    comm_stream = torch.cuda.Stream()
    comp_stream = torch.cuda.Stream()

    dist.barrier()
    # 预热
    for _ in range(5):
        with torch.cuda.stream(comm_stream):
            dist.all_to_all_single(output_tensor, input_tensor)
        with torch.cuda.stream(comp_stream):
            gemm_custom_grid(A, B, C, grid_dim_x, grid_dim_y)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(args.repeats):
        with torch.cuda.stream(comm_stream):
            dist.all_to_all_single(output_tensor, input_tensor)
        with torch.cuda.stream(comp_stream):
            gemm_custom_grid(A, B, C, grid_dim_x, grid_dim_y)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    dist.barrier()
    return (end_time - start_time) / args.repeats

def run_experiment(rank, world_size, args):
    """主实验函数"""
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    # --- 准备张量 ---
    # 通信张量 (注意：不再是列表)
    tensor_size = (args.comm_size, args.comm_size)
    comm_input = torch.randn(tensor_size, device=f'cuda:{local_rank}')
    comm_output = torch.empty(tensor_size, device=f'cuda:{local_rank}')
    
    # 计算张量
    M, K, N = args.comp_size, args.comp_size, args.comp_size
    A = torch.randn(M, K, device=f'cuda:{local_rank}')
    B = torch.randn(K, N, device=f'cuda:{local_rank}')
    C = torch.zeros(M, N, device=f'cuda:{local_rank}')

    # --- 运行基线测试 ---
    # 关键修正：直接传递参数，而不是打包成元组
    comm_baseline_time = timed_comm(comm_input, comm_output)
    
    if rank == 0:
        print("\n" + "="*50)
        print(f"实验设置: Comm Size={args.comm_size}, Comp Size={args.comp_size}, Repeats={args.repeats}")
        print(f"基线性能: 通信 = {comm_baseline_time*1000:.4f} ms")
        print("-" * 50)
        print(f"{'GEMM Grid Dim':<15} | {'Comp Baseline (ms)':<20} | {'Overlap Time (ms)':<20} | {'Slowdown Factor':<15}")
        print("-" * 80)

    for grid_dim in args.grid_dims:
        # 关键修正：直接传递参数
        comp_baseline_time = timed_comp(A, B, C, grid_dim, grid_dim)
        overlap_time = timed_overlap(comm_input, comm_output, A, B, C, grid_dim, grid_dim)
        
        ideal_overlap_time = max(comm_baseline_time, comm_baseline_time)
        slowdown_factor = overlap_time / ideal_overlap_time

        if rank == 0:
            print(f"{f'{grid_dim}x{grid_dim}':<15} | {comp_baseline_time*1000:<20.4f} | {overlap_time*1000:<20.4f} | {slowdown_factor:<15.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--comm-size', type=int, default=2048)
    parser.add_argument('--comp-size', type=int, default=2048)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--grid-dims', type=int, nargs='+', default=[1, 4, 8, 16, 32, 40, 56, 80])
    args = parser.parse_args()

    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    run_experiment(rank, world_size, args)