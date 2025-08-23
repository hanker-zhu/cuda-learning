import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import ctypes
import os
import threading
import time
from queue import Queue

# --- Utility Functions ---
def load_cuda_lib():
    """编译并加载CUDA共享库"""
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libkernels.so')
    # 如果库不存在，先编译
    if not os.path.exists(lib_path):
        print("Shared library not found. Compiling...")
        makefile_dir = os.path.dirname(lib_path)
        os.system(f"make -C {makefile_dir}")
        if not os.path.exists(lib_path):
            raise RuntimeError("Compilation failed. 'libkernels.so' not created.")
    
    # 加载共享库
    lib = ctypes.CDLL(lib_path)
    lib.launch_gemm.argtypes = [
        dim3, dim3, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p  # 新增 start_clk, end_clk
    ]
    lib.launch_vecadd.argtypes = [
        dim3, dim3, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p  # 新增 start_clk, end_clk
    ]
    print("CUDA library loaded successfully.")
    return lib

def print_smid_distribution(name, smid_array, num_blocks, threads_per_block):
    """打印SMID分布、线程块数和线程数"""
    unique_smids = np.unique(smid_array)
    smid_counts = {}
    for smid in smid_array:
        smid_counts[smid] = smid_counts.get(smid, 0) + 1
    
    print(f"{name} SMID Distribution:")
    print(f"  Total blocks: {num_blocks}")
    print(f"  Threads per block: {threads_per_block}")
    print(f"  Used SMIDs: {sorted(unique_smids)}")
    print(f"  SMID usage count: {dict(sorted(smid_counts.items()))}")
    print(f"  Number of different SMIDs: {len(unique_smids)}")
    
    # 检查是否有SMID冲突（多个block在同一个SMID上）
    conflicts = {smid: count for smid, count in smid_counts.items() if count > 1}
    if conflicts:
        print(f"  ⚠️  SMID conflicts detected: {conflicts}")
    else:
        print("  ✅ No SMID conflicts detected")

# --- 定义 dim3 结构体 ---
class dim3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]

# --- Kernel Execution Functions ---
def run_gemm(stream, repeats, warmup, M, N, K, d_A, d_B, d_C, d_smid, sync_barrier):
    """运行 GEMM kernel 并记录时间和block级别时间戳"""
    gemm_times = []
    lib = load_cuda_lib()
    gemm_blocks = dim3((N + 32 - 1) // 32, (M + 32 - 1) // 32, 1)
    gemm_threads = dim3(32, 32, 1)
    num_blocks = gemm_blocks.x * gemm_blocks.y

    # 分配block级别时间戳buffer
    d_start_clk = cuda.mem_alloc(num_blocks * 8)
    d_end_clk = cuda.mem_alloc(num_blocks * 8)
    h_start_clk = np.zeros(num_blocks, dtype=np.uint64)
    h_end_clk = np.zeros(num_blocks, dtype=np.uint64)

    # Warmup
    for _ in range(warmup):
        lib.launch_gemm(gemm_blocks, gemm_threads, int(d_A), int(d_B), int(d_C), M, N, K, int(d_smid), stream.handle,
                        int(d_start_clk), int(d_end_clk))
        stream.synchronize()

    print("GEMM warmup completed.")

    barrier_log = True
    for _ in range(repeats):
        if sync_barrier:
            if barrier_log:
                print("GEMM thread waiting at barrier...")
            sync_barrier.wait()
            if barrier_log:
                print("GEMM thread passed barrier.")
                barrier_log = False
        start_time = time.time()
        lib.launch_gemm(gemm_blocks, gemm_threads, int(d_A), int(d_B), int(d_C), M, N, K, int(d_smid), stream.handle,
                        int(d_start_clk), int(d_end_clk))
        stream.synchronize()
        end_time = time.time()
        gemm_times.append(end_time - start_time)

    # 输出block级别时间戳（仅repeats=1时详细打印）
    if repeats > 0 :
        cuda.memcpy_dtoh(h_start_clk, d_start_clk)
        cuda.memcpy_dtoh(h_end_clk, d_end_clk)
        print("GEMM block start/end clock64:")
        for i in range(num_blocks):
            print(f"  Block {i}: start={h_start_clk[i]}, end={h_end_clk[i]}, duration={h_end_clk[i]-h_start_clk[i]}")

    avg_time = sum(gemm_times) / len(gemm_times) * 1000  # 转换为毫秒
    print(f"GEMM kernel executed {repeats} times. Average time: {avg_time:.3f} ms.")
    return gemm_times

def run_vecadd(stream, repeats, warmup, VEC_SIZE, d_A, d_B, d_C, d_smid, sync_barrier):
    """运行 VecAdd kernel 并记录时间和block级别时间戳"""
    vecadd_times = []
    lib = load_cuda_lib()
    vecadd_blocks = dim3((VEC_SIZE + 1024 - 1) // 1024, 1, 1)
    vecadd_threads = dim3(1024, 1, 1)
    num_blocks = vecadd_blocks.x

    d_start_clk = cuda.mem_alloc(num_blocks * 8)
    d_end_clk = cuda.mem_alloc(num_blocks * 8)
    h_start_clk = np.zeros(num_blocks, dtype=np.uint64)
    h_end_clk = np.zeros(num_blocks, dtype=np.uint64)

    # Warmup
    for _ in range(warmup):
        lib.launch_vecadd(vecadd_blocks, vecadd_threads, int(d_A), int(d_B), int(d_C), VEC_SIZE, int(d_smid), stream.handle,
                          int(d_start_clk), int(d_end_clk))
        stream.synchronize()

    print("VecAdd warmup completed.")

    barrier_log = True
    for _ in range(repeats):
        if sync_barrier:
            if barrier_log:
                print("VecAdd thread waiting at barrier...")
            sync_barrier.wait()
            if barrier_log:
                print("VecAdd thread passed barrier.")
                barrier_log = False
        start_time = time.time()
        lib.launch_vecadd(vecadd_blocks, vecadd_threads, int(d_A), int(d_B), int(d_C), VEC_SIZE, int(d_smid), stream.handle,
                          int(d_start_clk), int(d_end_clk))
        stream.synchronize()
        end_time = time.time()
        vecadd_times.append(end_time - start_time)

    if repeats > 0:
        cuda.memcpy_dtoh(h_start_clk, d_start_clk)
        cuda.memcpy_dtoh(h_end_clk, d_end_clk)
        print("VecAdd block start/end clock64:")
        for i in range(num_blocks):
            print(f"  Block {i}: start={h_start_clk[i]}, end={h_end_clk[i]}, duration={h_end_clk[i]-h_start_clk[i]}")

    avg_time = sum(vecadd_times) / len(vecadd_times) * 1000  # 转换为毫秒
    print(f"VecAdd kernel executed {repeats} times. Average time: {avg_time:.3f} ms.")
    return vecadd_times

# --- Data Initialization Functions ---
def init_gemm_data(M, N, K):
    """初始化GEMM数据并分配GPU内存"""
    h_A = np.random.randn(M, K).astype(np.float32)
    h_B = np.random.randn(K, N).astype(np.float32)
    h_C = np.empty((M, N), dtype=np.float32)
    
    d_A = cuda.mem_alloc(h_A.nbytes)
    d_B = cuda.mem_alloc(h_B.nbytes)
    d_C = cuda.mem_alloc(h_C.nbytes)
    
    cuda.memcpy_htod(d_A, h_A)
    cuda.memcpy_htod(d_B, h_B)
    
    num_blocks = ((N + 32 - 1) // 32) * ((M + 32 - 1) // 32)
    d_smid = cuda.mem_alloc(num_blocks * 4)
    
    return d_A, d_B, d_C, d_smid

def init_vecadd_data(VEC_SIZE):
    """初始化VecAdd数据并分配GPU内存"""
    h_A = np.random.randn(VEC_SIZE).astype(np.float32)
    h_B = np.random.randn(VEC_SIZE).astype(np.float32)
    h_C = np.empty(VEC_SIZE, dtype=np.float32)
    
    d_A = cuda.mem_alloc(h_A.nbytes)
    d_B = cuda.mem_alloc(h_B.nbytes)
    d_C = cuda.mem_alloc(h_C.nbytes)
    
    cuda.memcpy_htod(d_A, h_A)
    cuda.memcpy_htod(d_B, h_B)
    
    num_blocks = (VEC_SIZE + 1024 - 1) // 1024
    d_smid = cuda.mem_alloc(num_blocks * 4)
    
    return d_A, d_B, d_C, d_smid

# --- Parallel Kernel Test ---
def test_parallel_kernels(M, N, K, VEC_SIZE, repeats=100, warmup=10, barrier=False):
    """测试并行运行的 GEMM 和 VecAdd kernel"""
    # 在主线程中预先分配数据，避免子线程上下文问题
    print("Initializing data in main thread...")
    
    # 计算 block 配置
    gemm_blocks_config = ((N + 32 - 1) // 32, (M + 32 - 1) // 32, 1)
    vecadd_blocks_config = ((VEC_SIZE + 1024 - 1) // 1024, 1, 1)
    gemm_num_blocks = gemm_blocks_config[0] * gemm_blocks_config[1] * gemm_blocks_config[2]
    vecadd_num_blocks = vecadd_blocks_config[0] * vecadd_blocks_config[1] * vecadd_blocks_config[2]
    
    # GEMM 数据分配（只分配内存，不初始化数据）
    d_A_gemm = cuda.mem_alloc(M * K * 4)  # float32 = 4 bytes
    d_B_gemm = cuda.mem_alloc(K * N * 4)
    d_C_gemm = cuda.mem_alloc(M * N * 4)
    
    # GEMM SMID 缓冲区
    d_smid_gemm_buffer = cuda.mem_alloc(gemm_num_blocks * 4)
    h_smid_gemm = np.zeros(gemm_num_blocks, dtype=np.int32)
    
    # VecAdd 数据分配（只分配内存，不初始化数据）
    d_A_vec = cuda.mem_alloc(VEC_SIZE * 4)
    d_B_vec = cuda.mem_alloc(VEC_SIZE * 4)
    d_C_vec = cuda.mem_alloc(VEC_SIZE * 4)
    
    # VecAdd SMID 缓冲区
    d_smid_vecadd_buffer = cuda.mem_alloc(vecadd_num_blocks * 4)
    h_smid_vecadd = np.zeros(vecadd_num_blocks, dtype=np.int32)
    
    sync_barrier = threading.Barrier(2) if barrier else None
    
    # 用于存储每轮的耗时
    gemm_time_queue = Queue()
    vecadd_time_queue = Queue()

    # 用于存储SMID信息的队列
    gemm_smid_queue = Queue()
    vecadd_smid_queue = Queue()
    
    # 用于线程间同步的事件
    data_ready_event = threading.Event()
    
    lib = load_cuda_lib()

    def gemm_thread_func():
        # 使用主线程分配的设备内存
        gemm_blocks = dim3(gemm_blocks_config[0], gemm_blocks_config[1], gemm_blocks_config[2])
        gemm_threads = dim3(32, 32, 1)
        
        print(f"GEMM: Grid size=({gemm_blocks.x}, {gemm_blocks.y}, {gemm_blocks.z}), Block size=({gemm_threads.x}, {gemm_threads.y}, {gemm_threads.z})")
        print(f"GEMM: Total blocks={gemm_blocks.x * gemm_blocks.y * gemm_blocks.z}")
        
        # Warmup - 等待数据准备
        for i in range(warmup):
            data_ready_event.wait()  # 等待主线程准备数据
            data_ready_event.clear()
            lib.launch_gemm(gemm_blocks, gemm_threads, int(d_A_gemm), int(d_B_gemm), int(d_C_gemm), M, N, K, int(d_smid_gemm_buffer), 0, 0, 0)
            cuda.Context.synchronize()
        print("GEMM warmup completed.")
        
        barrier_log = True
        for i in range(repeats):
            if sync_barrier:
                if barrier_log:
                    print("GEMM thread waiting at barrier...")
                sync_barrier.wait()
                if barrier_log:
                    print("GEMM thread passed barrier.")
                    barrier_log = False
            
            data_ready_event.wait()  # 等待主线程准备数据
            data_ready_event.clear()
            
            start_time = time.time()
            lib.launch_gemm(gemm_blocks, gemm_threads, int(d_A_gemm), int(d_B_gemm), int(d_C_gemm), M, N, K, int(d_smid_gemm_buffer), 0, 0, 0)
            cuda.Context.synchronize()
            end_time = time.time()
            gemm_time_queue.put(end_time - start_time)
            
            # 只在第一次运行时收集SMID信息
            if i == 0:
                cuda.memcpy_dtoh(h_smid_gemm, d_smid_gemm_buffer)
                gemm_smid_queue.put(h_smid_gemm.copy())

    def vecadd_thread_func():
        # 使用主线程分配的设备内存
        vecadd_blocks = dim3(vecadd_blocks_config[0], vecadd_blocks_config[1], vecadd_blocks_config[2])
        vecadd_threads = dim3(1024, 1, 1)
        
        print(f"VecAdd: Grid size=({vecadd_blocks.x}, {vecadd_blocks.y}, {vecadd_blocks.z}), Block size=({vecadd_threads.x}, {vecadd_threads.y}, {vecadd_threads.z})")
        print(f"VecAdd: Total blocks={vecadd_blocks.x * vecadd_blocks.y * vecadd_blocks.z}")
        
        # Warmup - 等待数据准备
        for i in range(warmup):
            data_ready_event.wait()  # 等待主线程准备数据
            data_ready_event.clear()
            lib.launch_vecadd(vecadd_blocks, vecadd_threads, int(d_A_vec), int(d_B_vec), int(d_C_vec), VEC_SIZE, int(d_smid_vecadd_buffer), 0, 0, 0)
            cuda.Context.synchronize()
        print("VecAdd warmup completed.")
        
        barrier_log = True
        for i in range(repeats):
            if sync_barrier:
                if barrier_log:
                    print("VecAdd thread waiting at barrier...")
                sync_barrier.wait()
                if barrier_log:
                    print("VecAdd thread passed barrier.")
                    barrier_log = False
            
            data_ready_event.wait()  # 等待主线程准备数据
            data_ready_event.clear()
            
            start_time = time.time()
            lib.launch_vecadd(vecadd_blocks, vecadd_threads, int(d_A_vec), int(d_B_vec), int(d_C_vec), VEC_SIZE, int(d_smid_vecadd_buffer), 0, 0, 0)
            cuda.Context.synchronize()
            end_time = time.time()
            vecadd_time_queue.put(end_time - start_time)
            
            # 只在第一次运行时收集SMID信息
            if i == 0:
                cuda.memcpy_dtoh(h_smid_vecadd, d_smid_vecadd_buffer)
                vecadd_smid_queue.put(h_smid_vecadd.copy())

    # 启动子线程
    gemm_thread = threading.Thread(target=gemm_thread_func)
    vecadd_thread = threading.Thread(target=vecadd_thread_func)
    gemm_thread.start()
    vecadd_thread.start()

    # 在主线程中生成数据并同步
    # Warmup 数据生成
    for i in range(warmup):
        h_A_gemm = np.random.randn(M, K).astype(np.float32)
        h_B_gemm = np.random.randn(K, N).astype(np.float32)
        cuda.memcpy_htod(d_A_gemm, h_A_gemm)
        cuda.memcpy_htod(d_B_gemm, h_B_gemm)
        
        h_A_vec = np.random.randn(VEC_SIZE).astype(np.float32)
        h_B_vec = np.random.randn(VEC_SIZE).astype(np.float32)
        cuda.memcpy_htod(d_A_vec, h_A_vec)
        cuda.memcpy_htod(d_B_vec, h_B_vec)
        
        data_ready_event.set()  # 通知子线程数据已准备好
        time.sleep(0.001)  # 给子线程一点时间处理
    
    # 测试数据生成
    for i in range(repeats):
        h_A_gemm = np.random.randn(M, K).astype(np.float32)
        h_B_gemm = np.random.randn(K, N).astype(np.float32)
        cuda.memcpy_htod(d_A_gemm, h_A_gemm)
        cuda.memcpy_htod(d_B_gemm, h_B_gemm)
        
        h_A_vec = np.random.randn(VEC_SIZE).astype(np.float32)
        h_B_vec = np.random.randn(VEC_SIZE).astype(np.float32)
        cuda.memcpy_htod(d_A_vec, h_A_vec)
        cuda.memcpy_htod(d_B_vec, h_B_vec)
        
        data_ready_event.set()  # 通知子线程数据已准备好
        time.sleep(0.001)  # 给子线程一点时间处理

    gemm_thread.join()
    vecadd_thread.join()

    print("Parallel kernel test completed.")

    # 分析SMID分布
    if not gemm_smid_queue.empty() and not vecadd_smid_queue.empty():
        gemm_smids = gemm_smid_queue.get()
        vecadd_smids = vecadd_smid_queue.get()
        
        print("\n--- SMID Analysis ---")
        print_smid_distribution("GEMM", gemm_smids, len(gemm_smids), 32*32)
        print_smid_distribution("VecAdd", vecadd_smids, len(vecadd_smids), 1024)
        
        # 检查两个kernel之间的SMID冲突
        gemm_unique = set(gemm_smids)
        vecadd_unique = set(vecadd_smids)
        shared_smids = gemm_unique.intersection(vecadd_unique)
        
        print(f"\n--- Cross-Kernel SMID Analysis ---")
        print(f"GEMM used SMIDs: {sorted(gemm_unique)}")
        print(f"VecAdd used SMIDs: {sorted(vecadd_unique)}")
        if shared_smids:
            print(f"⚠️  Shared SMIDs between kernels: {sorted(shared_smids)}")
        else:
            print("✅ No SMID sharing between kernels")

    # 计算平均时间
    avg_gemm_time = sum(gemm_time_queue.queue) / len(gemm_time_queue.queue) * 1000 if not gemm_time_queue.empty() else 0
    avg_vecadd_time = sum(vecadd_time_queue.queue) / len(vecadd_time_queue.queue) * 1000 if not vecadd_time_queue.empty() else 0

    return avg_gemm_time, avg_vecadd_time

def test_serial_kernels(M, N, K, VEC_SIZE, repeats=100, warmup=10):
    """测试串行运行的 GEMM 和 VecAdd kernel"""
    stream = cuda.Stream()

    # 数据内存分配（不初始化）
    d_A_gemm = cuda.mem_alloc(M * K * 4)
    d_B_gemm = cuda.mem_alloc(K * N * 4)
    d_C_gemm = cuda.mem_alloc(M * N * 4)

    d_A_vec = cuda.mem_alloc(VEC_SIZE * 4)
    d_B_vec = cuda.mem_alloc(VEC_SIZE * 4)
    d_C_vec = cuda.mem_alloc(VEC_SIZE * 4)

    lib = load_cuda_lib()

    # 添加block配置信息输出
    gemm_blocks = dim3((N + 32 - 1) // 32, (M + 32 - 1) // 32, 1)
    gemm_threads = dim3(32, 32, 1)
    vecadd_blocks = dim3((VEC_SIZE + 1024 - 1) // 1024, 1, 1)
    vecadd_threads = dim3(1024, 1, 1)
    
    print(f"GEMM: Grid size=({gemm_blocks.x}, {gemm_blocks.y}, {gemm_blocks.z}), Block size=({gemm_threads.x}, {gemm_threads.y}, {gemm_threads.z})")
    print(f"GEMM: Total blocks={gemm_blocks.x * gemm_blocks.y * gemm_blocks.z}")
    print(f"VecAdd: Grid size=({vecadd_blocks.x}, {vecadd_blocks.y}, {vecadd_blocks.z}), Block size=({vecadd_threads.x}, {vecadd_threads.y}, {vecadd_threads.z})")
    print(f"VecAdd: Total blocks={vecadd_blocks.x * vecadd_blocks.y * vecadd_blocks.z}")

    # 分配SMID缓冲区
    gemm_num_blocks = gemm_blocks.x * gemm_blocks.y * gemm_blocks.z
    vecadd_num_blocks = vecadd_blocks.x * vecadd_blocks.y * vecadd_blocks.z
    
    d_smid_gemm_buffer = cuda.mem_alloc(gemm_num_blocks * 4)
    d_smid_vecadd_buffer = cuda.mem_alloc(vecadd_num_blocks * 4)
    h_smid_gemm = np.zeros(gemm_num_blocks, dtype=np.int32)
    h_smid_vecadd = np.zeros(vecadd_num_blocks, dtype=np.int32)

    # Warmup - 每次生成新的随机数据
    for _ in range(warmup):
        h_A_gemm = np.random.randn(M, K).astype(np.float32)
        h_B_gemm = np.random.randn(K, N).astype(np.float32)
        cuda.memcpy_htod(d_A_gemm, h_A_gemm)
        cuda.memcpy_htod(d_B_gemm, h_B_gemm)
        
        h_A_vec = np.random.randn(VEC_SIZE).astype(np.float32)
        h_B_vec = np.random.randn(VEC_SIZE).astype(np.float32)
        cuda.memcpy_htod(d_A_vec, h_A_vec)
        cuda.memcpy_htod(d_B_vec, h_B_vec)
        
        lib.launch_gemm(gemm_blocks, gemm_threads,
                        int(d_A_gemm), int(d_B_gemm), int(d_C_gemm), M, N, K, int(d_smid_gemm_buffer), stream.handle, 0, 0)
        stream.synchronize()
        lib.launch_vecadd(vecadd_blocks, vecadd_threads,
                          int(d_A_vec), int(d_B_vec), int(d_C_vec), VEC_SIZE, int(d_smid_vecadd_buffer), stream.handle, 0, 0)
        stream.synchronize()

    print("Serial warmup completed.")

    # Run - 每次运行都生成新的随机数据
    gemm_times = []
    vecadd_times = []

    for i in range(repeats):
        # 为GEMM生成新的随机数据
        h_A_gemm = np.random.randn(M, K).astype(np.float32)
        h_B_gemm = np.random.randn(K, N).astype(np.float32)
        cuda.memcpy_htod(d_A_gemm, h_A_gemm)
        cuda.memcpy_htod(d_B_gemm, h_B_gemm)
        
        start_time = time.time()
        lib.launch_gemm(gemm_blocks, gemm_threads,
                        int(d_A_gemm), int(d_B_gemm), int(d_C_gemm), M, N, K, int(d_smid_gemm_buffer), stream.handle, 0, 0)
        stream.synchronize()
        end_time = time.time()
        gemm_times.append(end_time - start_time)

        # 为VecAdd生成新的随机数据
        h_A_vec = np.random.randn(VEC_SIZE).astype(np.float32)
        h_B_vec = np.random.randn(VEC_SIZE).astype(np.float32)
        cuda.memcpy_htod(d_A_vec, h_A_vec)
        cuda.memcpy_htod(d_B_vec, h_B_vec)
        
        start_time = time.time()
        lib.launch_vecadd(vecadd_blocks, vecadd_threads,
                          int(d_A_vec), int(d_B_vec), int(d_C_vec), VEC_SIZE, int(d_smid_vecadd_buffer), stream.handle, 0, 0)
        stream.synchronize()
        end_time = time.time()
        vecadd_times.append(end_time - start_time)
        
        # 只在第一次运行时收集SMID信息
        if i == 0:
            cuda.memcpy_dtoh(h_smid_gemm, d_smid_gemm_buffer)
            cuda.memcpy_dtoh(h_smid_vecadd, d_smid_vecadd_buffer)

    # 分析SMID分布
    print("\n--- SMID Analysis ---")
    print_smid_distribution("GEMM", h_smid_gemm, gemm_num_blocks, 32*32)
    print_smid_distribution("VecAdd", h_smid_vecadd, vecadd_num_blocks, 1024)

    avg_gemm_time = sum(gemm_times) / len(gemm_times) * 1000  # 转换为毫秒
    avg_vecadd_time = sum(vecadd_times) / len(vecadd_times) * 1000  # 转换为毫秒

    print(f"Serial GEMM kernel average time: {avg_gemm_time:.3f} ms.")
    print(f"Serial VecAdd kernel average time: {avg_vecadd_time:.3f} ms.")

    return avg_gemm_time, avg_vecadd_time

if __name__ == "__main__":
    print("\n--- Testing Serial Kernels ---")
    serial_gemm_time, serial_vecadd_time = test_serial_kernels(M=128, N=128, K=4096, VEC_SIZE=1024 * 24, repeats=100, warmup=10)

    print("\n--- Testing Parallel Kernels ---")
    parallel_gemm_time, parallel_vecadd_time = test_parallel_kernels(M=128, N=128, K=4096, VEC_SIZE=1024 * 24, repeats=100, warmup=10, barrier=True)

    print("\n--- Comparison ---")
    print(f"Serial GEMM kernel average time: {serial_gemm_time:.3f} ms.")
    print(f"Parallel GEMM kernel average time: {parallel_gemm_time:.3f} ms.")
    print(f"Serial VecAdd kernel average time: {serial_vecadd_time:.3f} ms.")
    print(f"Parallel VecAdd kernel average time: {parallel_vecadd_time:.3f} ms.")
