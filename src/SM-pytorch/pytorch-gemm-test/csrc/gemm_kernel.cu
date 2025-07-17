#include <cuda_runtime.h>

// 一个基础的、非优化的 GEMM CUDA 核函数
// 目标是消耗 SM 资源，而非追求极致性能
__global__ void basic_gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; ++i) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

// 新增：一个简单的向量加法 CUDA 核函数
__global__ void vector_add_kernel(const float* X, const float* Y, float* Z, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        Z[idx] = X[idx] + Y[idx];
    }
}

// 主机端包装函数，使用 extern "C" 防止 C++ 名称修饰
// 关键：grid_dim_x 和 grid_dim_y 参数允许我们从外部控制线程块数量
extern "C" {
void launch_gemm_kernel(float* A, float* B, float* C, int M, int N, int K, int grid_dim_x, int grid_dim_y) {
    // 为简单起见，固定线程块内的线程数量
    dim3 threads_per_block(16, 16);
    
    // 使用从 Python 传入的参数来定义Grid大小
    dim3 num_blocks(grid_dim_x, grid_dim_y);

    // 启动 CUDA 核函数
    basic_gemm_kernel<<<num_blocks, threads_per_block>>>(A, B, C, M, N, K);
}

// 修改：向量加法核函数的启动器，接受 grid_dim_x
void launch_vector_add_kernel(float* X, float* Y, float* Z, int size, int grid_dim_x) {
    int threads_per_block = 256;
    // 使用传入的 grid_dim_x 作为线程块数量
    dim3 num_blocks(grid_dim_x);
    vector_add_kernel<<<num_blocks, threads_per_block>>>(X, Y, Z, size);
}
}