#include <torch/extension.h>

// 声明我们将从 gemm_kernel.cu 中链接的函数
// 必须使用 extern "C" 以匹配定义
extern "C" {
void launch_gemm_kernel(float* A, float* B, float* C, int M, int N, int K, int grid_dim_x, int grid_dim_y);
}

// Python 可调用的包装函数
void gemm_custom_grid(torch::Tensor A, torch::Tensor B, torch::Tensor C, int grid_dim_x, int grid_dim_y) {
    // 检查张量是否在 CUDA 上且是连续的
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "所有张量必须在 CUDA 设备上");
    A = A.contiguous();
    B = B.contiguous();
    C = C.contiguous();

    // 获取矩阵维度
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // 调用我们的 CUDA 核函数启动器
    launch_gemm_kernel(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        grid_dim_x, grid_dim_y
    );
}

// 将 C++ 函数绑定到 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_custom_grid", &gemm_custom_grid, "使用自定义Grid维度的GEMM核函数");
}