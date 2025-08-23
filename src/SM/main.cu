#include <stdio.h>
#include <cuda.h>
#include <stdint.h>

/**
 * @brief 计算密集型核函数 (GEMM: C = A * B)
 * @param A 输入矩阵A
 * @param B 输入矩阵B
 * @param C 输出矩阵C
 * @param M 矩阵A的行数
 * @param N 矩阵B的列数
 * @param K 矩阵A的列数/B的行数
 * @param smid_store 用于存储每个线程块所在SMID的数组
 * @param start_clk 用于记录每个线程块开始时间的数组
 * @param end_clk 用于记录每个线程块结束时间的数组
 */
__global__ void gemm_kernel(const float *A, const float *B, float *C, int M, int N, int K, int *smid_store,
                            uint64_t *start_clk, uint64_t *end_clk) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 记录每个block的开始时间
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (start_clk) start_clk[blockIdx.y * gridDim.x + blockIdx.x] = clock64();
    }

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));
        smid_store[blockIdx.y * gridDim.x + blockIdx.x] = smid;
        if (end_clk) end_clk[blockIdx.y * gridDim.x + blockIdx.x] = clock64();
    }
}

/**
 * @brief 访存密集型核函数 (Vector Addition: C = A + B)
 * @param A 输入向量A
 * @param B 输入向量B
 * @param C 输出向量C
 * @param N 向量大小
 * @param smid_store 用于存储每个线程块所在SMID的数组
 * @param start_clk 用于记录每个线程块开始时间的数组
 * @param end_clk 用于记录每个线程块结束时间的数组
 */
__global__ void vecadd_kernel(const float *A, const float *B, float *C, int N, int *smid_store,
                              uint64_t *start_clk, uint64_t *end_clk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) {
        if (start_clk) start_clk[blockIdx.x] = clock64();
    }
    if (i < N) {
        C[i] = A[i] + B[i];
    }
    if (threadIdx.x == 0) {
        int smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));
        smid_store[blockIdx.x] = smid;
        if (end_clk) end_clk[blockIdx.x] = clock64();
    }
}

extern "C" {
    /**
     * @brief 启动GEMM核函数
     * @param numBlocks 网格维度
     * @param threadsPerBlock 每个线程块的线程数
     * @param A 输入矩阵A
     * @param B 输入矩阵B
     * @param C 输出矩阵C
     * @param M 矩阵A的行数
     * @param N 矩阵B的列数
     * @param K 矩阵A的列数/B的行数
     * @param smid_store 用于存储每个线程块所在SMID的数组
     * @param stream CUDA流
     * @param start_clk 用于记录每个线程块开始时间的数组
     * @param end_clk 用于记录每个线程块结束时间的数组
     */
    void launch_gemm(dim3 numBlocks, dim3 threadsPerBlock, const float *A, const float *B, float *C, int M, int N, int K,
                     int *smid_store, cudaStream_t stream,
                     uint64_t *start_clk = nullptr, uint64_t *end_clk = nullptr) {
        gemm_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(A, B, C, M, N, K, smid_store, start_clk, end_clk);
    }

    /**
     * @brief 启动Vector Add核函数
     * @param numBlocks 网格维度
     * @param threadsPerBlock 每个线程块的线程数
     * @param A 输入向量A
     * @param B 输入向量B
     * @param C 输出向量C
     * @param N 向量大小
     * @param smid_store 用于存储每个线程块所在SMID的数组
     * @param stream CUDA流
     * @param start_clk 用于记录每个线程块开始时间的数组
     * @param end_clk 用于记录每个线程块结束时间的数组
     */
    void launch_vecadd(dim3 numBlocks, dim3 threadsPerBlock, const float *A, const float *B, float *C, int N,
                       int *smid_store, cudaStream_t stream,
                       uint64_t *start_clk = nullptr, uint64_t *end_clk = nullptr) {
        vecadd_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(A, B, C, N, smid_store, start_clk, end_clk);
    }
}