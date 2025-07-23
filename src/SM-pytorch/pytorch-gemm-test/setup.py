import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取PyTorch路径用于设置库路径
pytorch_dir = os.path.dirname(torch.__file__)

# 设置CUDA_HOME环境变量，如果尚未设置
if 'CUDA_HOME' not in os.environ:
    cuda_path = '/usr/local/cuda'  # 默认路径
    if os.path.exists(cuda_path):
        os.environ['CUDA_HOME'] = cuda_path
        print(f"已设置CUDA_HOME={cuda_path}")

# 设置编译和链接参数
extra_compile_args = {
    'cxx': ['-std=c++14'],
    'nvcc': [
        '-std=c++14',
        '--expt-relaxed-constexpr',
        '-Xcompiler',
        '-fPIC'
    ]
}

# 添加运行时库路径
extra_link_args = [f'-Wl,-rpath,{pytorch_dir}/lib']

setup(
    name='gemm_test_ext',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='gemm_test_ext._C',  # 扩展模块的名称
            sources=[
                'csrc/bindings.cpp',
                'csrc/gemm_kernel.cu',
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            library_dirs=[f'{pytorch_dir}/lib'],
            runtime_library_dirs=[f'{pytorch_dir}/lib']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    verbose=True
)