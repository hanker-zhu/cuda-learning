import torch.distributed as dist
import torch
import time
import argparse
import threading
import os

def comm(stream, repeats, warmup, hang, dim1, dim2, dtype, fn, sync_barrier, barrier):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    M, N = dim1, dim2
    N //= world_size
    C = torch.randn(M, N, device=f'cuda:{local_rank}', dtype=dtype)
    input = [torch.randn(M, N, device=f'cuda:{local_rank}', dtype=dtype) for _ in range(world_size)]
    output = [torch.empty(M, N, device=f'cuda:{local_rank}', dtype=dtype) for _ in range(world_size)]
    total_elements = M * N * (world_size - 1)

    ## warmup
    event = torch.cuda.Event()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            if fn == 'all_reduce':
                dist.all_reduce(C)
            elif fn == 'all_gather':
                dist.all_gather(output, C)
            elif fn == 'all_to_all':
                dist.all_to_all(output, input)
            event.record(stream)
            event.synchronize()
    print(f"===========comm warmup {warmup} done=== rank: {rank}")
    sync_barrier.wait()

    ## run
    while True:
        times = []
        event = torch.cuda.Event()
        with torch.cuda.stream(stream):
            for _ in range(repeats):
                if barrier:
                    sync_barrier.wait()
                start_time = time.time()
                if fn == 'all_reduce':
                    dist.all_reduce(C)
                elif fn == 'all_gather':
                    dist.all_gather(output, C)
                elif fn == 'all_to_all':
                    dist.all_to_all(output, input)
                event.record(stream)
                event.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
        
        if rank == 0:
            # 计算统计量
            avg_time = sum(times) / len(times) * 1000      # 平均耗时（毫秒）
            min_time = min(times) * 1000                # 最小耗时（毫秒）
            max_time = max(times) * 1000                # 最大耗时（毫秒）

            total_elements *= (2 if dtype == torch.float32 else 1)
            # 打印结果
            print(f'''
COLL TEST ({fn}, worldsize {world_size}, {dtype}, {total_elements /1024/1024} MB):
- 最小耗时: {min_time:.3f} ms, 最大吞吐量: {total_elements / (min_time / 1000) / 1e9:.2f} GB/s
- 平均耗时: {avg_time:.3f} ms, 平均吞吐量: {total_elements / (avg_time / 1000) / 1e9:.2f} GB/s
- 最大耗时: {max_time:.3f} ms, 最小吞吐量: {total_elements / (max_time / 1000) / 1e9:.2f} GB/s
''')

        if not hang:
            break
    
def compute(stream, repeats, warmup, hang, M, N, K, dtype, sync_barrier, barrier):
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    # 计算矩阵
    A = torch.randn(M, K, device=f'cuda:{local_rank}', dtype=dtype)
    B = torch.randn(K, N, device=f'cuda:{local_rank}', dtype=dtype)

    ## warmup
    event = torch.cuda.Event()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            _ = torch.matmul(A, B)
            event.record(stream)
            event.synchronize()
    print(f"===========comp warmup {warmup} done=== rank: {rank}")
    sync_barrier.wait()
    
    ## run
    while True:
        # 计时
        times = []
        event = torch.cuda.Event()
        with torch.cuda.stream(stream): 
            for _ in range(repeats):
                if barrier:
                    sync_barrier.wait()
                start_time = time.time()
                _ = torch.matmul(A, B)
                event.record(stream)
                event.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
        # torch.cuda.synchronize()
        if rank == 0:
            # 计算统计量
            avg_time = sum(times) / len(times) * 1000      # 平均耗时（毫秒）
            min_time = min(times) * 1000                # 最小耗时（毫秒）
            max_time = max(times) * 1000                # 最大耗时（毫秒）
            
            # 计算FLOPS（理论值：2*M*N*K / 时间）
            total_flops = 2 * M * N * K
            # 打印结果
            print(f'''
GEMM TEST [{M}x{K}] * [{K}x{N}] ({dtype}):
- 最小耗时: {min_time:.3f} ms, 最大吞吐量: {total_flops / (min_time / 1000) / 1e12:.2f} TFLOPS
- 平均耗时: {avg_time:.3f} ms, 平均吞吐量: {total_flops / (avg_time / 1000) / 1e12:.2f} TFLOPS
- 最大耗时: {max_time:.3f} ms, 最小吞吐量: {total_flops / (max_time / 1000) / 1e12:.2f} TFLOPS
''')

        if not hang:
            break



def test_gemm_performance(M, K, N, dim1, dim2, dtype=torch.float32, repeats=100, warmup=0, hang=False, barrier=False, fn='all_gather'):
    thread_count = 0
    if M > 0:
        thread_count += 1
    if dim2 > 0:
        thread_count += 1
    sync_barrier = threading.Barrier(thread_count)  # 同步屏障，确保两个线程同时触发操作


    local_rank = int(os.environ['LOCAL_RANK'])

    if dim2 > 0:
        comm_stream = torch.cuda.Stream(device=local_rank, priority=-2) # device=rank, 
        comm_thread = threading.Thread(
            target=comm,
            args=(comm_stream, repeats, warmup, hang, dim1, dim2, dtype, fn, sync_barrier, barrier)
        )
        comm_thread.start()
    if M > 0:
        compute_stream = torch.cuda.Stream(device=local_rank, priority=-1) # device=rank, 
        compute_thread = threading.Thread(
            target=compute, 
            args=(compute_stream, repeats, warmup, hang, M, N, K, dtype, sync_barrier, barrier)
        )
        compute_thread.start()

    while hang:
        time.sleep(1)  # 保持主线程存活



if __name__ == "__main__":
    # 初始化分布式进程组
    dist.init_process_group(backend='nccl', init_method='env://')
    
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-rank', '--local_rank', type=int)  # 由torch.distributed.launch自动传入
    parser.add_argument('-m', type=int, default=4096, help='矩阵A的行数')
    parser.add_argument('-k', type=int, default=4096, help='矩阵A的列数 and 矩阵B的行数')
    parser.add_argument('-n', type=int, default=4096, help='矩阵B的列数')
    parser.add_argument('--comp-size', type=str, default='')
    parser.add_argument('--comm-size', type=int, default=0)
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--hang', action='store_true')
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--no-barrier', '--no_barrier', action='store_true')
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--fn', type=str, choices=['all_reduce', 'all_gather', 'all_to_all'], default='all_gather')
    args = parser.parse_args()
    
    M, K, N = args.m, args.k, args.n
    if args.comp_size == '0':
        M, K, N = 0, 0, 0
    elif args.comp_size == '0.125':
        M, K, N = 128, 1024, 1024
    elif args.comp_size == '1':
        M, K, N = 1024, 1024, 1024
    elif args.comp_size == '4':
        M, K, N = 1024, 1024, 4096
    elif args.comp_size == '16':
        M, K, N = 1024, 4096, 4096
    elif args.comp_size == '64':
        M, K, N = 4096, 4096, 4096
    elif args.comp_size == '200':
        M, K, N = 2048, 5120, 20480
    
    dim1, dim2 = 1024, 1024
    dim2 *= args.comm_size

    # 运行测试函数
    test_gemm_performance(
        M=M, K=K, N=N, dim1=dim1, dim2=dim2,
        dtype=torch.float16 if args.fp16 else torch.float32,
        repeats=args.repeats,
        warmup=args.warmup,
        hang=args.hang,
        barrier=not args.no_barrier,
        fn=args.fn,
    )
