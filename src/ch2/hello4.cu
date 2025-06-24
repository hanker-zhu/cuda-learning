#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_from_gpu() {
    printf("Hello from GPU from block (%d, %d, %d) and thread (%d, %d, %d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
    
    // sleep random time to simulate work
    unsigned long long int sleep_time = 1000000 * 1000;
    printf("Sleeping for %llu seconds\n", sleep_time);
    unsigned long long int start = clock64();
    while (clock64() - start < sleep_time) {
        // Busy wait for the specified time 
    }
    printf("Finished sleeping in block (%d, %d, %d) and thread (%d, %d, %d)  start clock is (%llu)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
            start
        );
}

int main(void) {
    dim3 grid(1, 2, 1);
    dim3 block(1, 1, 3);

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