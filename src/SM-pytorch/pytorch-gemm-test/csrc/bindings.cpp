#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>

// 声明我们将从 gemm_kernel.cu 中链接的函数
extern "C" {
    void launch_gemm_kernel(float* A, float* B, float* C, int M, int N, int K, int grid_dim_x, int grid_dim_y, int* smid_out, float sleep_ms);
    void launch_vector_add_kernel(float* X, float* Y, float* Z, int size, int grid_dim_x, int* smid_out, float sleep_ms);
}

// Python 可调用的包装函数
torch::Tensor gemm_custom_grid(torch::Tensor A, torch::Tensor B, torch::Tensor C, int grid_dim_x, int grid_dim_y) {
    // 检查张量是否在 CUDA 上且是连续的
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "所有张量必须在 CUDA 设备上");
    A = A.contiguous();
    B = B.contiguous();
    C = C.contiguous();

    // 获取矩阵维度
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    // 设置设备保护
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    
    // 计算总线程块数量
    int total_blocks = grid_dim_x * grid_dim_y;
    
    // 创建一个新的张量来存储SMID值 - 每个线程块一个值
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(C.device());
    auto smid_tensor = torch::empty({total_blocks}, options);

    // 调用我们的 CUDA 核函数启动器
    // 暂时设置sleep_ms为0
    float sleep_ms = 0.0f;
    launch_gemm_kernel(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        grid_dim_x, grid_dim_y,
        smid_tensor.data_ptr<int>(),
        sleep_ms
    );
    
    return smid_tensor;
}

// 向量加法的 Python 包装函数
torch::Tensor vector_add(torch::Tensor X, torch::Tensor Y, torch::Tensor Z, int grid_dim_x) {
    TORCH_CHECK(X.is_cuda() && Y.is_cuda() && Z.is_cuda(), "所有张量必须在 CUDA 设备上");
    X = X.contiguous();
    Y = Y.contiguous();
    Z = Z.contiguous();
    TORCH_CHECK(X.size(0) == Y.size(0) && Y.size(0) == Z.size(0), "所有向量必须有相同的尺寸");
    
    // 设置设备保护
    const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
    
    // 创建一个新的张量来存储SMID值 - 每个线程块一个值
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(Z.device());
    auto smid_tensor = torch::empty({grid_dim_x}, options);

    // 暂时设置sleep_ms为0
    float sleep_ms = 0.0f;
    launch_vector_add_kernel(
        X.data_ptr<float>(),
        Y.data_ptr<float>(),
        Z.data_ptr<float>(),
        X.size(0),
        grid_dim_x,
        smid_tensor.data_ptr<int>(),
        sleep_ms
    );
        
    return smid_tensor;
}

// 将 C++ 函数绑定到 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_custom_grid", &gemm_custom_grid, "使用自定义Grid维度的GEMM核函数，返回SMID");
    m.def("vector_add", &vector_add, "一个简单的向量加法核函数，使用自定义Grid维度，返回SMID");
}