#include <stdio.h>

/**
 * @brief 计算密集型核函数 (GEMM: C = A * B)
 * @param A 输入矩阵A
 * @param B 输入矩阵B
 * @param C 输出矩阵C
 * @param M 矩阵A的行数
 * @param N 矩阵B的列数
 * @param K 矩阵A的列数/B的行数
 * @param smid_store 用于存储每个线程块所在SMID的数组
 */
__global__ void gemm_kernel(const float *A, const float *B, float *C, int M, int N, int K, int *smid_store) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

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
    }
}

/**
 * @brief 访存密集型核函数 (Vector Addition: C = A + B)
 * @param A 输入向量A
 * @param B 输入向量B
 * @param C 输出向量C
 * @param N 向量大小
 * @param smid_store 用于存储每个线程块所在SMID的数组
 */
__global__ void vecadd_kernel(const float *A, const float *B, float *C, int N, int *smid_store) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }

    if (threadIdx.x == 0) {
        int smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));
        smid_store[blockIdx.x] = smid;
    }
}

extern "C" {
    /**
     * @brief 启动GEMM核函数
     */
    void launch_gemm(dim3 numBlocks, dim3 threadsPerBlock, const float *A, const float *B, float *C, int M, int N, int K, int *smid_store, cudaStream_t stream) {
        gemm_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(A, B, C, M, N, K, smid_store);
    }

    /**
     * @brief 启动Vector Add核函数
     */
    void launch_vecadd(dim3 numBlocks, dim3 threadsPerBlock, const float *A, const float *B, float *C, int N, int *smid_store, cudaStream_t stream) {
        vecadd_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(A, B, C, N, smid_store);
    }
}