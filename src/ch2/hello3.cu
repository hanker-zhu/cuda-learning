#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_from_gpu() {
    printf("Hello from GPU from block %d, thread %d!\n", blockIdx.x, threadIdx.x);
    // Note: blockIdx.x is the block index in the grid, and threadIdx.x
}

int main(void) {
    hello_from_gpu<<<4, 4>>>();
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    printf("Hello from CPU\n");
    return 0;
}