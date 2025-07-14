import torch
import time
import argparse
# from gemm_test import gemm_extension  # Assuming the C++ extension is named gemm_extension
from gemm_test_ext import _C as gemm_extension

def run_gemm_test(M, K, N, block_size):
    # Create random matrices
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    C = torch.zeros(M, N, dtype=torch.float32, device='cuda')

    # Pass block_size for both block_size_x and block_size_y
    gemm_extension.gemm(A, B, C, block_size, block_size)

    # Measure performance
    start_time = time.time()
    for _ in range(100):  # Run multiple times for averaging
        # Pass block_size for both block_size_x and block_size_y
        gemm_extension.gemm(A, B, C, block_size, block_size)
    torch.cuda.synchronize()  # Ensure all kernels are finished
    end_time = time.time()

    avg_time = (end_time - start_time) / 100  # Average time per operation
    print(f'Average time for GEMM with block size {block_size}: {avg_time * 1000:.3f} ms')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GEMM Performance Test')
    parser.add_argument('--M', type=int, default=1024, help='Number of rows in matrix A')
    parser.add_argument('--K', type=int, default=1024, help='Number of columns in matrix A and rows in matrix B')
    parser.add_argument('--N', type=int, default=1024, help='Number of columns in matrix B')
    parser.add_argument('--block-size', type=int, default=16, help='Thread block size for GEMM operation')
    
    args = parser.parse_args()
    
    run_gemm_test(args.M, args.K, args.N, args.block_size)