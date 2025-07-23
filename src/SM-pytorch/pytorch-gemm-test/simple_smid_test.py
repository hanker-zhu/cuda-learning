import torch
import numpy as np

print("=== 简单SMID测试 ===")

# 初始化CUDA设备
device = torch.device('cuda:0')
print(f"使用设备: {torch.cuda.get_device_name(device)}")
print(f"设备SM数量: {torch.cuda.get_device_properties(device).multi_processor_count}")

# 导入自定义CUDA扩展
try:
    from gemm_test_ext._C import gemm_custom_grid, vector_add
    print("成功导入CUDA扩展")
except ImportError as e:
    print(f"导入错误: {e}")
    import sys
    sys.exit(1)

# 准备小型输入数据
M, N, K = 64, 64, 64
A = torch.randn(M, K, device=device)
B = torch.randn(K, N, device=device)
C = torch.zeros(M, N, device=device)

vec_size = 1024
X = torch.randn(vec_size, device=device)
Y = torch.randn(vec_size, device=device)
Z = torch.zeros(vec_size, device=device)

# 测试GEMM
print("\n--- 测试GEMM SMID收集 ---")
blocks_x, blocks_y = 2, 2
print(f"使用线程块: {blocks_x}x{blocks_y}")

smids = gemm_custom_grid(A, B, C, blocks_x, blocks_y)
print(f"返回的张量类型: {type(smids)}")

if not torch.is_tensor(smids):
    print("错误: 未返回张量")
else:
    print(f"张量形状: {smids.shape}")
    print(f"张量内容: {smids}")
    
    # 检查是否有非零值
    non_zeros = (smids > 0).sum().item()
    print(f"非零值数量: {non_zeros}/{smids.numel()}")
    
    if non_zeros > 0:
        unique_smids = torch.unique(smids[smids > 0])
        print(f"唯一的SMID值: {unique_smids.cpu().numpy()}")
    else:
        print("没有检测到SMID值")

# 测试向量加法
print("\n--- 测试向量加法SMID收集 ---")
blocks = 4
print(f"使用线程块: {blocks}")

smids = vector_add(X, Y, Z, blocks)
print(f"返回的张量类型: {type(smids)}")

if not torch.is_tensor(smids):
    print("错误: 未返回张量")
else:
    print(f"张量形状: {smids.shape}")
    print(f"张量内容: {smids}")
    
    # 检查是否有非零值
    non_zeros = (smids > 0).sum().item()
    print(f"非零值数量: {non_zeros}/{smids.numel()}")
    
    if non_zeros > 0:
        unique_smids = torch.unique(smids[smids > 0])
        print(f"唯一的SMID值: {unique_smids.cpu().numpy()}")
    else:
        print("没有检测到SMID值")

print("\n测试完成")
