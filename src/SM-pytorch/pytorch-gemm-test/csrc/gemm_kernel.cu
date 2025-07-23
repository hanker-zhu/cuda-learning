#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// 定义获取SMID的内联函数 - 直接使用PTX指令
__device__ __forceinline__ int get_smid() {
    int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// 非常简单的GEMM核函数，用于测试SM分配
__global__ void basic_gemm_kernel(const float* A, const float* B, float* C, 
                                 int M, int N, int K, int* smid_out, float sleep_ms=0.0f, int clock_rate=1000000) {
    // 获取线程索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程块只记录一次SMID (使用第一个线程)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // 计算线程块索引
        int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
        
        // 获取当前SM的ID
        int smid = get_smid();
        
        // 记录SMID到输出数组 - 确保值大于0
        smid_out[block_idx] = smid + 1;  // 加1确保0不被当作无效值
    }
    
    // 执行实际的GEMM计算
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
    
    // 添加可选的人工延迟，模拟长时间运行的内核
    if (sleep_ms > 0.0f) {
        // 通过时钟计数实现延迟
        clock_t start_clock = clock();
        clock_t clock_offset = (clock_t)(sleep_ms * (clock_rate / 1000.0f));
        
        // 用计算替代纯等待，防止编译器优化
        float dummy_result = 0.0f;
        int iterations = clock_offset / 100;
        for (int i = 0; i < iterations; i++) {
            dummy_result += sinf(dummy_result + 0.1f) * cosf(dummy_result + 0.2f);
        }
        
        // 使用dummy_result防止编译器优化
        if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && dummy_result > 1e10) {
            printf("Dummy: %f\n", dummy_result);
        }
    }
}

// 简单的向量加法核函数
__global__ void vector_add_kernel(const float* X, const float* Y, float* Z, 
                                 int size, int* smid_out, float sleep_ms=0.0f, int clock_rate=1000000) {
    // 获取线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程块只记录一次SMID (使用第一个线程)
    if (threadIdx.x == 0) {
        // 获取当前SM的ID
        int smid = get_smid();
        
        // 记录SMID到输出数组 - 确保值大于0
        smid_out[blockIdx.x] = smid + 1;  // 加1确保0不被当作无效值
    }
    
    // 执行实际的向量加法计算
    if (idx < size) {
        Z[idx] = X[idx] + Y[idx];
    }
    
    // 添加可选的人工延迟，模拟长时间运行的内核
    if (sleep_ms > 0.0f) {
        // 通过时钟计数实现延迟
        clock_t start_clock = clock();
        clock_t clock_offset = (clock_t)(sleep_ms * (clock_rate / 1000.0f));
        
        // 用计算替代纯等待，防止编译器优化
        float dummy_result = 0.0f;
        int iterations = clock_offset / 100;
        for (int i = 0; i < iterations; i++) {
            dummy_result += sinf(dummy_result + 0.1f) * cosf(dummy_result + 0.2f);
        }
        
        // 使用dummy_result防止编译器优化
        if (threadIdx.x == 0 && blockIdx.x == 0 && dummy_result > 1e10) {
            printf("Dummy: %f\n", dummy_result);
        }
    }
}

// 主机端包装函数
extern "C" {

void launch_gemm_kernel(float* A, float* B, float* C, int M, int N, int K, 
                        int grid_dim_x, int grid_dim_y, int* smid_out, float sleep_ms=0.0f) {
    // 设置线程块大小
    dim3 threads_per_block(16, 16);
    dim3 num_blocks(grid_dim_x, grid_dim_y);
    
    // 获取设备属性以获取时钟频率
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int clock_rate = prop.clockRate;
    
    // 启动内核
    basic_gemm_kernel<<<num_blocks, threads_per_block>>>(A, B, C, M, N, K, smid_out, sleep_ms, clock_rate);
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GEMM内核启动错误: %s\n", cudaGetErrorString(err));
    }
}

void launch_vector_add_kernel(float* X, float* Y, float* Z, int size, 
                             int grid_dim_x, int* smid_out, float sleep_ms=0.0f) {
    // 设置线程块大小
    int threads_per_block = 256;
    
    // 获取设备属性以获取时钟频率
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int clock_rate = prop.clockRate;
    
    // 启动内核
    vector_add_kernel<<<grid_dim_x, threads_per_block>>>(X, Y, Z, size, smid_out, sleep_ms, clock_rate);
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("向量加法内核启动错误: %s\n", cudaGetErrorString(err));
    }
}

}