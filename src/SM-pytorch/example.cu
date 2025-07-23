#include <stdio.h>

cudaDeviceProp gpu_info() {
    printf("============CUDA Device Information:==============\n");
    int device_count;
    cudaGetDeviceCount(&device_count);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM count: %d\n", prop.multiProcessorCount);
    printf("  Max warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
    printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads dim: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Max shared memory per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("  Max registers per block: %d\n", prop.regsPerBlock);
    printf("  Max registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("  Clock rate: %d Hz\n", prop.clockRate);
    printf("  Memory clock rate: %d Hz\n", prop.memoryClockRate);
    printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
    printf("  Max blocks Per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    return prop;
}

template<int REGS_COUNT>
__global__ void kernel_example(int kernel_index, int *smid_store, int smem_size, float seconds, clock_t clock_rate) {
    clock_t t0 = clock64();
    
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    // if (kernel_index == -1 && bid < 2) {
    //     if (tid == 0) smid_store[bid] = -1;
    //     return;
    // }
    if (tid == 0) smid_store[bid] = smid;

    extern __shared__ int smem[];
    // const int REGS_COUNT = 256;
    const int smem_count = smem_size / sizeof(int);
    int regs[REGS_COUNT];
    
    #pragma unroll REGS_COUNT
    for (int i = 0; i < smem_count; i++) {
        regs[i%REGS_COUNT] += smem[i];
    }

    for (int i = 0; i < REGS_COUNT && i < smem_count; i++) {
        smem[i] = regs[i];
    }
    
    clock_t t1 = clock64();
    while ((t1 - t0) / (clock_rate * 1000.0f) < seconds) {
        t1 = clock64();
    }
    if (bid == 0 && tid == 0) {
        printf("kernel %d finished\n", kernel_index);
    }   
}

cudaEvent_t get_event(cudaStream_t stream) {
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream);
    return event;
    // Some Synchronize Ways
    // cudaStreamWaitEvent(streams[i], event, 0);
    // cudaEventSynchronize(event);
    // cudaStreamSynchronize(streams[i]);
}

int main() {
    // gpu info 
    cudaDeviceProp prop = gpu_info();
    clock_t clock_rate = prop.clockRate;

    // initialize launch parameters
    const int num_kernel = 3;
    int num_blocks[num_kernel] = {20, 20, 20};
    int num_threads[num_kernel] = {32, 32, 32};
    int smem_size[num_kernel] = {16, 8, 8}; // KB
    int regs_count[num_kernel] = {32, 32, 32};
    bool with_barrier[num_kernel] = {0, 0, 0};
    float seconds[num_kernel] = {1.0, 1.0, 1.0};
    for (int i = 0; i < num_kernel; i++) {
        smem_size[i] *= 1024; // KB
    }

    // print launch args
    printf("============Launch Args:==============\n");
    printf("id\ttb\tthread\tsmem\tregs\tbarrier\n");
    for (int i = 0; i < num_kernel; i++ ) {
        printf("%d\t%d\t%d\t%d\t%d\t%d\n", i, num_blocks[i], num_threads[i], smem_size[i], regs_count[i], with_barrier[i]);
    }

    // print max TB
    for (int i = 0; i < num_kernel; i++) {
        // printf("MAX tbs supported by sm:            %d\n", prop.maxBlocksPerMultiProcessor);
        // printf("MAX tbs calculate by threads:       %d\n", prop.maxThreadsPerMultiProcessor / num_threads[0]);
        // printf("MAX tbs calculate by shared memory: %zu\n", prop.sharedMemPerMultiprocessor / smem_size[0]);
        // printf("MAX tbs calculate by regs:          %d\n", prop.regsPerMultiprocessor / (num_threads[0] * regs_count[0]));
        int a = prop.maxBlocksPerMultiProcessor;
        a = min(a, prop.maxThreadsPerMultiProcessor / num_threads[i]);
        a = min(a, int(prop.sharedMemPerMultiprocessor / smem_size[i]));
        a = min(a, prop.regsPerMultiprocessor / (num_threads[i] * regs_count[i]));
        printf("Max TBs for kernel %d : %d \n", i, a);
    }
    cudaStream_t streams[num_kernel];
    for (int i = 0; i < num_kernel; i++) {
        cudaStreamCreate(&streams[i]);
    }
    cudaEvent_t start_event;
    cudaEvent_t end_event[num_kernel];

    int all_blocks = 0;
    for (int i = 0; i < num_kernel; i++) {
        all_blocks += num_blocks[i];
    }
    int accumulate_blocks = 0;

    // init smid_store
    int *smid_store;
    cudaMallocManaged(&smid_store, all_blocks * sizeof(int));

    // launch kernels
    printf("============Kernel Launch:==============\n");
    printf("Number of kernels: %d\n", num_kernel);
    cudaDeviceSynchronize();
    clock_t t0 = clock();
    start_event = get_event(nullptr);
    for (int i = 0; i < num_kernel; i++) {
        dim3 Grid(num_blocks[i], 1, 1);
        dim3 Block(num_threads[i], 1, 1);
        if (regs_count[i] == 512) {
            kernel_example<512><<<Grid, Block, smem_size[i], streams[i]>>>
            (i, smid_store+accumulate_blocks, smem_size[i], seconds[i], clock_rate);
        } 
        else if (regs_count[i] == 256) {
            kernel_example<256><<<Grid, Block, smem_size[i], streams[i]>>>
            (i, smid_store+accumulate_blocks, smem_size[i], seconds[i], clock_rate);
        }
        else if (regs_count[i] == 128) {
            kernel_example<128><<<Grid, Block, smem_size[i], streams[i]>>>
            (i, smid_store+accumulate_blocks, smem_size[i], seconds[i], clock_rate);
        }
        else if (regs_count[i] == 64) {
            kernel_example<64><<<Grid, Block, smem_size[i], streams[i]>>>
            (i, smid_store+accumulate_blocks, smem_size[i], seconds[i], clock_rate);
        }
        else if (regs_count[i] == 32) {
            kernel_example<32><<<Grid, Block, smem_size[i], streams[i]>>>
            (i, smid_store+accumulate_blocks, smem_size[i], seconds[i], clock_rate);
        }
        end_event[i] = get_event(streams[i]);
        if (with_barrier[i]) {
            cudaStreamSynchronize(streams[i]);
        }
        accumulate_blocks += num_blocks[i];
    }
    cudaDeviceSynchronize();
    clock_t t1 = clock();
    float elapsed_time = (float)(t1 - t0) / CLOCKS_PER_SEC;
    printf("Total elapsed time for kernel launches: %.2f seconds\n", elapsed_time);
    printf("Kernels not be overlapped: %.0f\n", elapsed_time / seconds[0]);
    for (int i = 0; i < num_kernel; i++) {
        float event_time;
        cudaEventElapsedTime(&event_time, start_event, end_event[i]);
        printf("Kernel %d end at: %.2f s\n", i, event_time/1000);
    }

    // print kernel information
    printf("============Kernel Schedule:==============\n");
    accumulate_blocks = 0;
    for (int i = 0; i < num_kernel; i++) {
        for (int j = 0; j < num_blocks[i]; j++) {
            printf("Kernel %d, Block %d, SMID %d\n", i, j, smid_store[j+accumulate_blocks]);
        }
        accumulate_blocks += num_blocks[i];
    }
    
    // cleanup
    for (int i = 0; i < num_kernel; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(smid_store);
    
    return 0;
}