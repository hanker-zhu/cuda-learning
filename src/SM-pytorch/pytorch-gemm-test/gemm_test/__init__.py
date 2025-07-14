# Import the C++ extension module
from gemm_test_ext import _C as gemm_extension

# You can optionally expose specific functions from the extension module here
# For example, if your C++ extension has a function called 'gemm':
# from .gemm_extension import gemm

__all__ = ['gemm_extension']