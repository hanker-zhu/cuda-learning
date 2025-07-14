#include <iostream>
#include <vector>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <utility>
#include <random>
#include <chrono>

/*
================================================================================
 HOW TO COMPILE:
--------------------------------------------------------------------------------
 nvcc -o nano_sim nano_sim.cu -std=c++11
 ./nano_sim
================================================================================
*/

// Macro for robust CUDA error checking
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " in file " << __FILE__ \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// =============================================================================
// KERNELS (Provided by user)
// =============================================================================

// compute-bound kernel : matrix multiplication
// NOTE: This is a naive implementation for demonstration. A real-world GEMM
// would use shared memory, tiling, and other optimizations.
__global__ void gemm_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// memory-bound kernel : vector addition with stride access pattern
__global__ void vec_add_kernel(const float *A, const float *B, float *C, long long N, int stride = 1) {
    long long idx = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Random data initialization function
void init_random_data(std::vector<float>& data, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (auto& val : data) {
        val = dis(gen);
    }
}

// =============================================================================
// MAIN SIMULATION LOGIC
// =============================================================================

int main() {
    std::cout << "Starting NanoFlow Algorithm Simulation..." << std::endl;

    // --- 0. Query GPU properties ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "\n[GPU INFO] Device: " << prop.name << std::endl;
    std::cout << "  - Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  - L2 Cache Size: " << prop.l2CacheSize / 1024 / 1024 << " MB" << std::endl;
    std::cout << "  - Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;

    // --- 1. Setup Problem Size and Configuration ---
    // 增大问题规模，确保超出共享内存和L1缓存大小
    const int M = 4096;  // 4K x 4K matrices
    const int N = 4096;
    const int K = 4096;
    const long long VEC_SIZE = 256LL * 1024 * 1024;  // 256M elements = 1GB data per vector

    std::cout << "\n[PROBLEM SIZE]" << std::endl;
    std::cout << "  - Matrix dimensions: " << M << " x " << N << " x " << K << std::endl;
    std::cout << "  - Matrix memory: " << (M * K + K * N + M * N) * sizeof(float) / 1024 / 1024 << " MB" << std::endl;
    std::cout << "  - Vector size: " << VEC_SIZE << " elements" << std::endl;
    std::cout << "  - Vector memory per array: " << VEC_SIZE * sizeof(float) / 1024 / 1024 << " MB" << std::endl;

    // Kernel launch configurations - 更大的block配置
    const dim3 block_size_gemm(32, 32);  // 1024 threads per block
    const dim3 grid_size_gemm((N + block_size_gemm.x - 1) / block_size_gemm.x, 
                              (M + block_size_gemm.y - 1) / block_size_gemm.y);

    const int threads_per_block_vec = 1024;  // 最大线程数
    const long long blocks_needed = (VEC_SIZE + threads_per_block_vec - 1) / threads_per_block_vec;
    const dim3 grid_size_vec(std::min(blocks_needed, (long long)65535));  // 限制在GPU最大grid size内

    std::cout << "\n[KERNEL CONFIG]" << std::endl;
    std::cout << "  - GEMM grid: (" << grid_size_gemm.x << ", " << grid_size_gemm.y << ")" << std::endl;
    std::cout << "  - GEMM block: (" << block_size_gemm.x << ", " << block_size_gemm.y << ")" << std::endl;
    std::cout << "  - VecAdd grid: " << grid_size_vec.x << std::endl;
    std::cout << "  - VecAdd block: " << threads_per_block_vec << std::endl;

    // --- 2. Allocate and Initialize Memory ---
    std::cout << "\n[INFO] Allocating memory and initializing random data..." << std::endl;

    // Host memory with random initialization
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_D(VEC_SIZE);
    std::vector<float> h_E(VEC_SIZE);
    std::vector<float> h_F(VEC_SIZE, 0.0f);

    // 初始化随机数据
    std::cout << "  - Initializing random matrix data..." << std::endl;
    init_random_data(h_A, -1.0f, 1.0f);
    init_random_data(h_B, -1.0f, 1.0f);
    
    std::cout << "  - Initializing random vector data..." << std::endl;
    init_random_data(h_D, -1.0f, 1.0f);
    init_random_data(h_E, -1.0f, 1.0f);

    // Device memory
    float *d_A, *d_B, *d_C, *d_D, *d_E, *d_F;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, h_C.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D, h_D.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_E, h_E.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_F, h_F.size() * sizeof(float)));

    // Copy data from host to device
    std::cout << "  - Copying data to GPU..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D, h_D.data(), h_D.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E, h_E.data(), h_E.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- 3. Step 1: Baseline Performance (Individual Execution) ---
    std::cout << "\n--- Step 1: Measuring Baseline Performance ---" << std::endl;
    
    // 多次测量取平均值
    const int num_warmup = 5;
    const int num_runs = 10;
    
    // Warmup
    for (int i = 0; i < num_warmup; i++) {
        gemm_kernel<<<grid_size_gemm, block_size_gemm>>>(d_A, d_B, d_C, M, N, K);
        vec_add_kernel<<<grid_size_vec, threads_per_block_vec>>>(d_D, d_E, d_F, VEC_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    float time_gemm_base = 0.0f, time_vec_add_base = 0.0f;

    // 测量GEMM性能
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        gemm_kernel<<<grid_size_gemm, block_size_gemm>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        time_gemm_base += time_ms;
    }
    time_gemm_base /= num_runs;

    // 测量VecAdd性能
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        vec_add_kernel<<<grid_size_vec, threads_per_block_vec>>>(d_D, d_E, d_F, VEC_SIZE);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        time_vec_add_base += time_ms;
    }
    time_vec_add_base /= num_runs;

    std::cout << "  - GEMM Kernel (alone): " << std::fixed << std::setprecision(4) << time_gemm_base << " ms" << std::endl;
    std::cout << "  - VecAdd Kernel (alone): " << std::fixed << std::setprecision(4) << time_vec_add_base << " ms" << std::endl;

    // 计算理论性能
    double gemm_flops = 2.0 * M * N * K;
    double gemm_gflops = (gemm_flops / 1e9) / (time_gemm_base / 1000.0);
    double vecadd_bandwidth = (3.0 * VEC_SIZE * sizeof(float) / 1e9) / (time_vec_add_base / 1000.0);
    
    std::cout << "  - GEMM Performance: " << std::fixed << std::setprecision(2) << gemm_gflops << " GFLOPS" << std::endl;
    std::cout << "  - VecAdd Bandwidth: " << std::fixed << std::setprecision(2) << vecadd_bandwidth << " GB/s" << std::endl;

    // --- 4. Step 2: Interference Profiling (Simulating NanoFlow's Analysis) ---
    std::cout << "\n--- Step 2: Interference Profiling (Finding Optimal R) ---" << std::endl;
    std::cout << "This simulates creating the Resource-Performance table by varying the\n"
              << "resource allocation 'R' for the compute-bound kernel (GEMM).\n" << std::endl;

    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    std::vector<std::pair<float, float>> profiling_results;
    std::cout << std::setw(15) << "R_gemm" << std::setw(25) << "Concurrent Time (ms)" << std::setw(20) << "Speedup" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (float R = 1.0f; R >= 0.1f; R -= 0.1f) {
        dim3 current_grid_gemm = grid_size_gemm;
        current_grid_gemm.x = std::max(1u, (unsigned int)(grid_size_gemm.x * R));
        current_grid_gemm.y = std::max(1u, (unsigned int)(grid_size_gemm.y * R));

        float time_concurrent = 0.0f;
        
        // 多次测量取平均值
        for (int i = 0; i < num_runs; i++) {
            // 每次使用不同的数据偏移，避免缓存效应
            int offset = i * 1024;  // 不同的内存偏移
            
            CUDA_CHECK(cudaEventRecord(start));
            gemm_kernel<<<current_grid_gemm, block_size_gemm, 0, stream1>>>(d_A, d_B, d_C, M, N, K);
            vec_add_kernel<<<grid_size_vec, threads_per_block_vec, 0, stream2>>>(
                d_D + (offset % (VEC_SIZE/2)), d_E + (offset % (VEC_SIZE/2)), d_F + (offset % (VEC_SIZE/2)), 
                VEC_SIZE - (offset % (VEC_SIZE/2)));
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
            time_concurrent += time_ms;
        }
        time_concurrent /= num_runs;
        
        float baseline_sum = time_gemm_base + time_vec_add_base;
        float speedup = baseline_sum / time_concurrent;
        
        profiling_results.push_back({R, time_concurrent});
        std::cout << std::fixed << std::setprecision(2) << std::setw(15) << R
                  << std::fixed << std::setprecision(4) << std::setw(25) << time_concurrent 
                  << std::fixed << std::setprecision(2) << std::setw(20) << speedup << std::endl;
    }

    auto best_result_it = std::min_element(profiling_results.begin(), profiling_results.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

    float optimal_R = best_result_it->first;
    float best_concurrent_time = best_result_it->second;
    
    std::cout << "\n[RESULT] Optimal R found: " << std::fixed << std::setprecision(2) << optimal_R 
              << ", which yields the best concurrent time of " << std::fixed << std::setprecision(4) << best_concurrent_time << " ms." << std::endl;

    // --- 5. Step 3: Final Comparison ---
    std::cout << "\n--- Step 3: Final Performance Comparison ---" << std::endl;

    // Strategy A: Sequential Execution
    float time_sequential = 0.0f;
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        gemm_kernel<<<grid_size_gemm, block_size_gemm>>>(d_A, d_B, d_C, M, N, K);
        vec_add_kernel<<<grid_size_vec, threads_per_block_vec>>>(d_D, d_E, d_F, VEC_SIZE);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        time_sequential += time_ms;
    }
    time_sequential /= num_runs;
    std::cout << "  - Strategy A (Sequential):      " << std::fixed << std::setprecision(4) << time_sequential << " ms" << std::endl;
    
    // Strategy B: Naive Parallel Execution
    float time_naive_parallel = 0.0f;
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        gemm_kernel<<<grid_size_gemm, block_size_gemm, 0, stream1>>>(d_A, d_B, d_C, M, N, K);
        vec_add_kernel<<<grid_size_vec, threads_per_block_vec, 0, stream2>>>(d_D, d_E, d_F, VEC_SIZE);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        time_naive_parallel += time_ms;
    }
    time_naive_parallel /= num_runs;
    std::cout << "  - Strategy B (Naive Parallel):    " << std::fixed << std::setprecision(4) << time_naive_parallel << " ms" << std::endl;

    // Strategy C: NanoFlow-style Optimized Parallel Execution
    dim3 optimal_grid_gemm = grid_size_gemm;
    optimal_grid_gemm.x = std::max(1u, (unsigned int)(grid_size_gemm.x * optimal_R));
    optimal_grid_gemm.y = std::max(1u, (unsigned int)(grid_size_gemm.y * optimal_R));
    
    float time_nanoflow = 0.0f;
    for (int i = 0; i < num_runs; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        gemm_kernel<<<optimal_grid_gemm, block_size_gemm, 0, stream1>>>(d_A, d_B, d_C, M, N, K);
        vec_add_kernel<<<grid_size_vec, threads_per_block_vec, 0, stream2>>>(d_D, d_E, d_F, VEC_SIZE);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        time_nanoflow += time_ms;
    }
    time_nanoflow /= num_runs;
    std::cout << "  - Strategy C (NanoFlow Optimal):" << std::fixed << std::setprecision(4) << time_nanoflow << " ms" << std::endl;

    // --- 6. Conclusion ---
    std::cout << "\n--- Conclusion ---" << std::endl;
    std::cout << "NanoFlow-style optimization provided a " << std::fixed << std::setprecision(2)
              << time_sequential / time_nanoflow << "x speedup over sequential execution." << std::endl;
    std::cout << "NanoFlow vs Naive Parallel: " << std::fixed << std::setprecision(2)
              << time_naive_parallel / time_nanoflow << "x speedup." << std::endl;

    // --- 7. Cleanup ---
    std::cout << "\n[INFO] Cleaning up resources..." << std::endl;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_E));
    CUDA_CHECK(cudaFree(d_F));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    return 0;
}