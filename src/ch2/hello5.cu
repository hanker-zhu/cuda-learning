#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_from_gpu() {
    // 线程按 threadIdx.x 从大到小输出
    for (int i = blockDim.x - 1; i >= 0; --i) {
        if (threadIdx.x == i) {
            printf("Hello from GPU from block (%d, %d, %d) and thread (%d, %d, %d)\n",
                   blockIdx.x, blockIdx.y, blockIdx.z,
                   threadIdx.x, threadIdx.y, threadIdx.z);
        }
        __syncthreads();
    }
}

int main(void) {
    dim3 grid(1, 2, 1);
    dim3 block(3, 1, 1);

    printf("Launching kernel with %d blocks and %d threads per block\n", 
        grid.x * grid.y * grid.z, block.x * block.y * block.z);
    printf("Hello from CPU\n");

    // Launch the kernel
    hello_from_gpu<<<grid, block>>>();

    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));


    printf("Hello from CPU\n");
    return 0;
}