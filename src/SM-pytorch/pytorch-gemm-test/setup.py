from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)