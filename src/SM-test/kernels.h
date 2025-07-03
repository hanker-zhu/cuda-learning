#ifndef KERNERLS_H
#define KERNERLS_H

// compute-bound kernel : matrix multiplication
__global__ void gemm_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++ k) {
            sum += A[row * K + k] * B[k *N + col];
        }

        C[row * N + col] = sum;
    }
}

// memory-bound kernel : vector addition
__global__ void vec_add_kernel(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}   

#endif // KERNERLS_H    