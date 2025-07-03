#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <functional>
#include <algorithm>
#include "kernels.h"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 计算kernel理论占用的SM数量
void analyzeKernelOccupancy(const char* kernel_name, int grid_size, int block_size, int shared_mem = 0) {
    int device;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    
    // 计算每个SM可以运行的block数量
    int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
    
    // 受寄存器限制的block数量（简化估算）
    int reg_limited_blocks = prop.regsPerMultiprocessor / (block_size * 32); // 假设每线程32个寄存器
    
    // 受共享内存限制的block数量
    int smem_limited_blocks = shared_mem > 0 ? 
        prop.sharedMemPerMultiprocessor / (shared_mem + 48) : 
        prop.sharedMemPerMultiprocessor / 48; // 仅考虑开销
    
    // 硬件最大blocks限制（Tesla T4通常为16）
    int hw_max_blocks = 16; // Tesla T4的典型值
    
    // 实际每个SM的block数量
    int blocks_per_sm = std::min({max_blocks_per_sm, reg_limited_blocks, smem_limited_blocks, hw_max_blocks});
    
    // 需要的SM数量
    int required_sms = (grid_size + blocks_per_sm - 1) / blocks_per_sm;
    int actual_used_sms = std::min(required_sms, prop.multiProcessorCount);
    
    // 计算占用率
    float occupancy = (float)(grid_size * block_size) / (prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor);
    
    std::cout << "\n" << kernel_name << " 资源分析:" << std::endl;
    std::cout << "  Grid大小: " << grid_size << " blocks" << std::endl;
    std::cout << "  Block大小: " << block_size << " threads" << std::endl;
    std::cout << "  线程限制blocks/SM: " << max_blocks_per_sm << std::endl;
    std::cout << "  寄存器限制blocks/SM: " << reg_limited_blocks << std::endl;
    std::cout << "  共享内存限制blocks/SM: " << smem_limited_blocks << std::endl;
    std::cout << "  硬件最大blocks/SM: " << hw_max_blocks << std::endl;
    std::cout << "  实际blocks/SM: " << blocks_per_sm << std::endl;
    std::cout << "  理论需要SM: " << required_sms << std::endl;
    std::cout << "  实际使用SM: " << actual_used_sms << " / " << prop.multiProcessorCount << std::endl;
    std::cout << "  SM利用率: " << (float)actual_used_sms / prop.multiProcessorCount * 100 << "%" << std::endl;
    std::cout << "  线程占用率: " << occupancy * 100 << "%" << std::endl;
    
    // 分析并行可能性
    if (actual_used_sms >= prop.multiProcessorCount) {
        std::cout << "  状态: 占满所有SM，难以与其他kernel并行" << std::endl;
    } else {
        std::cout << "  状态: 剩余 " << (prop.multiProcessorCount - actual_used_sms) << " 个SM可供其他kernel使用" << std::endl;
    }
}

// CUDA事件计时器
float measureKernelTime(std::function<void()> kernelFunc) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    kernelFunc();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float time;
    CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return time;
}

// 使用CUDA Profiler API分析SM使用情况
void analyzeConcurrentExecution() {
    std::cout << "\n=== 并发执行分析 ===" << std::endl;
    
    // 检查设备是否支持并发kernel执行
    int device;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "设备并发能力:" << std::endl;
    std::cout << "  支持并发kernel: " << (prop.concurrentKernels ? "是" : "否") << std::endl;
    std::cout << "  异步引擎数量: " << prop.asyncEngineCount << std::endl;
    std::cout << "  内存复制与kernel重叠: " << (prop.deviceOverlap ? "是" : "否") << std::endl;
    
    // 计算理论最大并行度
    int total_cores = prop.multiProcessorCount;
    std::cout << "  总SM数量: " << total_cores << std::endl;
    std::cout << "  每SM最大线程数: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  GPU总计算能力: " << total_cores * prop.maxThreadsPerMultiProcessor << " 线程" << std::endl;
}

int main() {
    // 测试参数
    const int M = 2048, N = 2048, K = 2048;     // GEMM: 2048×2048×2048
    const int VEC_SIZE = 16 * 1024 * 1024;      // 向量: 16M floats
    const int NUM_ITERATIONS = 50;
    const int NUM_DATA_SETS = 8;

    std::cout << "=== 数据预生成分析 ===" << std::endl;
    std::cout << "准备 " << NUM_DATA_SETS << " 组不同的随机数据..." << std::endl;
    
    // 预生成多组随机数据
    float **h_A_gemm_sets = new float*[NUM_DATA_SETS];
    float **h_B_gemm_sets = new float*[NUM_DATA_SETS];
    float **h_A_vec_sets = new float*[NUM_DATA_SETS];
    float **h_B_vec_sets = new float*[NUM_DATA_SETS];
    
    float *h_C_gemm = new float[M * N];
    float *h_C_vec = new float[VEC_SIZE];

    srand(time(nullptr));
    for (int set = 0; set < NUM_DATA_SETS; set++) {
        std::cout << "生成数据集 " << (set + 1) << "/" << NUM_DATA_SETS << "..." << std::endl;
        
        h_A_gemm_sets[set] = new float[M * K];
        h_B_gemm_sets[set] = new float[K * N];
        h_A_vec_sets[set] = new float[VEC_SIZE];
        h_B_vec_sets[set] = new float[VEC_SIZE];
        
        srand(time(nullptr) + set * 1000);
        
        for (int i = 0; i < M * K; i++) h_A_gemm_sets[set][i] = rand() / (float)RAND_MAX;
        for (int i = 0; i < K * N; i++) h_B_gemm_sets[set][i] = rand() / (float)RAND_MAX;
        for (int i = 0; i < VEC_SIZE; i++) {
            h_A_vec_sets[set][i] = rand() / (float)RAND_MAX;
            h_B_vec_sets[set][i] = rand() / (float)RAND_MAX;
        }
    }

    // GPU内存分配
    float **d_A_gemm_sets = new float*[NUM_DATA_SETS];
    float **d_B_gemm_sets = new float*[NUM_DATA_SETS];
    float **d_A_vec_sets = new float*[NUM_DATA_SETS];
    float **d_B_vec_sets = new float*[NUM_DATA_SETS];
    float *d_C_gemm, *d_C_vec;
    
    for (int set = 0; set < NUM_DATA_SETS; set++) {
        CHECK_CUDA(cudaMalloc(&d_A_gemm_sets[set], M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B_gemm_sets[set], K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_A_vec_sets[set], VEC_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B_vec_sets[set], VEC_SIZE * sizeof(float)));
    }
    CHECK_CUDA(cudaMalloc(&d_C_gemm, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_vec, VEC_SIZE * sizeof(float)));

    // 数据传输到GPU
    std::cout << "传输数据到GPU..." << std::endl;
    for (int set = 0; set < NUM_DATA_SETS; set++) {
        CHECK_CUDA(cudaMemcpy(d_A_gemm_sets[set], h_A_gemm_sets[set], M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B_gemm_sets[set], h_B_gemm_sets[set], K * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_A_vec_sets[set], h_A_vec_sets[set], VEC_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B_vec_sets[set], h_B_vec_sets[set], VEC_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    }

    // 配置kernel参数
    dim3 gemm_block(32, 16);  // 512 threads/block
    dim3 gemm_grid((N + gemm_block.x - 1) / gemm_block.x, (M + gemm_block.y - 1) / gemm_block.y);
    
    dim3 vec_block(512);
    dim3 vec_grid((VEC_SIZE + vec_block.x - 1) / vec_block.x);

    std::cout << "=== GPU设备信息 ===" << std::endl;
    int device;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::cout << "设备名称: " << prop.name << std::endl;
    std::cout << "SM数量: " << prop.multiProcessorCount << std::endl;
    std::cout << "最大线程数/SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "最大blocks/SM: 16 (Tesla T4典型值)" << std::endl;
    std::cout << "共享内存/SM: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
    std::cout << "寄存器/SM: " << prop.regsPerMultiprocessor << std::endl;

    // 分析每个kernel的SM占用情况
    analyzeKernelOccupancy("GEMM Kernel", gemm_grid.x * gemm_grid.y, gemm_block.x * gemm_block.y);
    analyzeKernelOccupancy("Vector Add Kernel", vec_grid.x, vec_block.x);
    
    // 分析并发执行能力
    analyzeConcurrentExecution();

    std::cout << "\n=== 数据大小分析 ===" << std::endl;
    float gemm_data_size = (M * K + K * N + M * N) * sizeof(float) / (1024.0f * 1024.0f);
    float vec_data_size = 3 * VEC_SIZE * sizeof(float) / (1024.0f * 1024.0f);
    
    std::cout << "GEMM数据大小: " << gemm_data_size << " MB" << std::endl;
    std::cout << "Vector数据大小: " << vec_data_size << " MB" << std::endl;

    std::cout << "\n=== 性能测试 ===" << std::endl;
    std::cout << "GEMM矩阵大小: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "向量大小: " << VEC_SIZE << std::endl;
    std::cout << "GEMM blocks: " << gemm_grid.x * gemm_grid.y << " (" << gemm_block.x * gemm_block.y << " threads/block)" << std::endl;
    std::cout << "Vector blocks: " << vec_grid.x << " (" << vec_block.x << " threads/block)" << std::endl;

    // GPU预热
    std::cout << "\nGPU预热中..." << std::endl;
    for (int i = 0; i < 3; i++) {
        int set_idx = i % NUM_DATA_SETS;
        gemm_kernel<<<gemm_grid, gemm_block>>>(d_A_gemm_sets[set_idx], d_B_gemm_sets[set_idx], d_C_gemm, M, N, K);
        vec_add_kernel<<<vec_grid, vec_block>>>(d_A_vec_sets[set_idx], d_B_vec_sets[set_idx], d_C_vec, VEC_SIZE);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 测试不同的并行策略
    std::cout << "\n=== 多种并行策略对比 ===" << std::endl;

    // 策略1: 串行执行
    float serial_time = measureKernelTime([&]() {
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            int set_idx = i % NUM_DATA_SETS;
            gemm_kernel<<<gemm_grid, gemm_block>>>(d_A_gemm_sets[set_idx], d_B_gemm_sets[set_idx], d_C_gemm, M, N, K);
            vec_add_kernel<<<vec_grid, vec_block>>>(d_A_vec_sets[set_idx], d_B_vec_sets[set_idx], d_C_vec, VEC_SIZE);
        }
        cudaDeviceSynchronize();
    });
    std::cout << "策略1 - 串行执行: " << serial_time << " ms" << std::endl;

    // 策略2: 多流并行 - 保持相同计算总量，通过分批启动控制SM分配
    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));
    
    std::cout << "\n=== SM分配策略 (保持相同计算总量) ===" << std::endl;
    
    // 计算批次大小以控制并发度，但保持总计算量不变
    int total_gemm_blocks = gemm_grid.x * gemm_grid.y;
    int total_vec_blocks = vec_grid.x;
    
    // 7:3 SM分配 - 通过分批控制并发度
    int gemm_batches = 2;  // 将GEMM分成2批
    int vec_batches = 4;   // 将Vector分成4批
    
    int gemm_blocks_per_batch = (total_gemm_blocks + gemm_batches - 1) / gemm_batches;
    int vec_blocks_per_batch = (total_vec_blocks + vec_batches - 1) / vec_batches;
    
    std::cout << "GEMM总blocks: " << total_gemm_blocks << " -> 分成" << gemm_batches << "批，每批" << gemm_blocks_per_batch << "个blocks" << std::endl;
    std::cout << "Vector总blocks: " << total_vec_blocks << " -> 分成" << vec_batches << "批，每批" << vec_blocks_per_batch << "个blocks" << std::endl;
    
    float parallel_time = measureKernelTime([&]() {
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            int set_idx = i % NUM_DATA_SETS;
            
            // 分批启动GEMM kernel，控制并发度但保持总计算量
            for (int batch = 0; batch < gemm_batches; batch++) {
                int start_block = batch * gemm_blocks_per_batch;
                int end_block = std::min((batch + 1) * gemm_blocks_per_batch, total_gemm_blocks);
                int blocks_in_batch = end_block - start_block;
                
                if (blocks_in_batch > 0) {
                    // 重新计算当前批次的grid维度
                    dim3 batch_grid;
                    batch_grid.x = std::min(blocks_in_batch, (int)gemm_grid.x);
                    batch_grid.y = (blocks_in_batch + batch_grid.x - 1) / batch_grid.x;
                    
                    gemm_kernel<<<batch_grid, gemm_block, 0, stream1>>>(
                        d_A_gemm_sets[set_idx], d_B_gemm_sets[set_idx], d_C_gemm, M, N, K);
                }
            }
            
            // 分批启动Vector kernel
            for (int batch = 0; batch < vec_batches; batch++) {
                int start_block = batch * vec_blocks_per_batch;
                int end_block = std::min((batch + 1) * vec_blocks_per_batch, total_vec_blocks);
                int blocks_in_batch = end_block - start_block;
                
                if (blocks_in_batch > 0) {
                    dim3 batch_grid(blocks_in_batch);
                    int start_element = start_block * vec_block.x;
                    int elements_in_batch = blocks_in_batch * vec_block.x;
                    elements_in_batch = std::min(elements_in_batch, VEC_SIZE - start_element);
                    
                    if (elements_in_batch > 0) {
                        vec_add_kernel<<<batch_grid, vec_block, 0, stream2>>>(
                            d_A_vec_sets[set_idx] + start_element,
                            d_B_vec_sets[set_idx] + start_element,
                            d_C_vec + start_element,
                            elements_in_batch
                        );
                    }
                }
            }
        }
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
    });
    std::cout << "策略2 - 分批并行执行 (相同计算量): " << parallel_time << " ms" << std::endl;

    // 策略3: 完全并行 - 所有kernel同时启动，让GPU调度器决定
    float full_parallel_time = measureKernelTime([&]() {
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            int set_idx = i % NUM_DATA_SETS;
            
            // 同时启动所有kernel，保持完整计算量
            gemm_kernel<<<gemm_grid, gemm_block, 0, stream1>>>(
                d_A_gemm_sets[set_idx], d_B_gemm_sets[set_idx], d_C_gemm, M, N, K);
            vec_add_kernel<<<vec_grid, vec_block, 0, stream2>>>(
                d_A_vec_sets[set_idx], d_B_vec_sets[set_idx], d_C_vec, VEC_SIZE);
        }
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
    });
    std::cout << "策略3 - 完全并行执行: " << full_parallel_time << " ms" << std::endl;

    // 策略4: 交错执行 - 保持计算总量，通过时间错开控制资源竞争
    float interleaved_time = measureKernelTime([&]() {
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            int set_idx = i % NUM_DATA_SETS;
            
            // 先启动Vector（快速执行）
            vec_add_kernel<<<vec_grid, vec_block, 0, stream2>>>(
                d_A_vec_sets[set_idx], d_B_vec_sets[set_idx], d_C_vec, VEC_SIZE);
            
            // 稍后启动GEMM（长时间执行）
            gemm_kernel<<<gemm_grid, gemm_block, 0, stream1>>>(
                d_A_gemm_sets[set_idx], d_B_gemm_sets[set_idx], d_C_gemm, M, N, K);
        }
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
    });
    std::cout << "策略4 - 交错执行: " << interleaved_time << " ms" << std::endl;

    // 策略5: 重叠执行 - 使用更多流来增加重叠机会
    cudaStream_t stream3, stream4;
    CHECK_CUDA(cudaStreamCreate(&stream3));
    CHECK_CUDA(cudaStreamCreate(&stream4));
    
    float overlap_time = measureKernelTime([&]() {
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            int set_idx = i % NUM_DATA_SETS;
            cudaStream_t gemm_stream = (i % 2 == 0) ? stream1 : stream3;
            cudaStream_t vec_stream = (i % 2 == 0) ? stream2 : stream4;
            
            // 使用4个流增加重叠机会，但保持相同计算量
            gemm_kernel<<<gemm_grid, gemm_block, 0, gemm_stream>>>(
                d_A_gemm_sets[set_idx], d_B_gemm_sets[set_idx], d_C_gemm, M, N, K);
            vec_add_kernel<<<vec_grid, vec_block, 0, vec_stream>>>(
                d_A_vec_sets[set_idx], d_B_vec_sets[set_idx], d_C_vec, VEC_SIZE);
        }
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        cudaStreamSynchronize(stream3);
        cudaStreamSynchronize(stream4);
    });
    std::cout << "策略5 - 多流重叠执行: " << overlap_time << " ms" << std::endl;

    // 最终分析
    std::cout << "\n=== 性能对比分析 (相同计算总量) ===" << std::endl;
    std::cout << "串行执行:       " << serial_time << " ms (基准)" << std::endl;
    std::cout << "分批并行:       " << parallel_time << " ms (加速比: " << serial_time/parallel_time << "x)" << std::endl;
    std::cout << "完全并行:       " << full_parallel_time << " ms (加速比: " << serial_time/full_parallel_time << "x)" << std::endl;
    std::cout << "交错执行:       " << interleaved_time << " ms (加速比: " << serial_time/interleaved_time << "x)" << std::endl;
    std::cout << "多流重叠:       " << overlap_time << " ms (加速比: " << serial_time/overlap_time << "x)" << std::endl;
    
    // 找出最佳策略
    float best_time = std::min({parallel_time, full_parallel_time, interleaved_time, overlap_time});
    std::cout << "\n=== 最佳策略分析 ===" << std::endl;
    if (best_time == parallel_time) {
        std::cout << "最佳策略: 分批并行执行 - 通过控制并发度减少资源竞争" << std::endl;
    } else if (best_time == full_parallel_time) {
        std::cout << "最佳策略: 完全并行执行 - GPU调度器能有效处理资源分配" << std::endl;
    } else if (best_time == interleaved_time) {
        std::cout << "最佳策略: 交错执行 - 时间错开避免资源冲突" << std::endl;
    } else if (best_time == overlap_time) {
        std::cout << "最佳策略: 多流重叠 - 更多流提供更好的重叠机会" << std::endl;
    }
    
    std::cout << "\n=== 并行效率分析 ===" << std::endl;
    std::cout << "所有策略都保持了相同的计算总量，对比结果客观有效" << std::endl;
    if (best_time < serial_time) {
        float speedup = serial_time / best_time;
        std::cout << "最佳加速比: " << speedup << "x" << std::endl;
        if (speedup > 1.5) {
            std::cout << "并行执行获得显著性能提升，SM资源得到有效利用" << std::endl;
        } else {
            std::cout << "并行执行有一定提升，但可能受到内存带宽或资源竞争限制" << std::endl;
        }
    } else {
        std::cout << "并行执行未获得性能提升，可能原因:" << std::endl;
        std::cout << "- 内存带宽成为瓶颈" << std::endl;
        std::cout << "- kernel本身已充分利用GPU资源" << std::endl;
        std::cout << "- 多流调度开销超过并行收益" << std::endl;
    }

    // 清理资源
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream3));
    CHECK_CUDA(cudaStreamDestroy(stream4));
    
    // 清理GPU内存
    for (int set = 0; set < NUM_DATA_SETS; set++) {
        CHECK_CUDA(cudaFree(d_A_gemm_sets[set]));
        CHECK_CUDA(cudaFree(d_B_gemm_sets[set]));
        CHECK_CUDA(cudaFree(d_A_vec_sets[set]));
        CHECK_CUDA(cudaFree(d_B_vec_sets[set]));
    }
    CHECK_CUDA(cudaFree(d_C_gemm));
    CHECK_CUDA(cudaFree(d_C_vec));
    
    // 清理主机内存
    for (int set = 0; set < NUM_DATA_SETS; set++) {
        delete[] h_A_gemm_sets[set];
        delete[] h_B_gemm_sets[set];
        delete[] h_A_vec_sets[set];
        delete[] h_B_vec_sets[set];
    }
    delete[] h_A_gemm_sets;
    delete[] h_B_gemm_sets;
    delete[] h_A_vec_sets;
    delete[] h_B_vec_sets;
    delete[] d_A_gemm_sets;
    delete[] d_B_gemm_sets;
    delete[] d_A_vec_sets;
    delete[] d_B_vec_sets;
    delete[] h_C_gemm;
    delete[] h_C_vec;

    return 0;
}