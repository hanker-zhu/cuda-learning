import torch
import numpy as np
from gemm_test_ext._C import gemm_custom_grid, vector_add

def test_smid_collection():
    """测试SMID收集功能是否正常工作"""
    print("===== 测试SMID收集功能 =====")
    
    # 设置设备
    device = 'cuda:0'
    torch.cuda.set_device(device)
    
    # 创建测试输入
    M, K, N = 1024, 1024, 1024
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    C = torch.zeros(M, N, device=device)
    
    vec_size = 1024 * 1024
    X = torch.randn(vec_size, device=device)
    Y = torch.randn(vec_size, device=device)
    Z = torch.zeros(vec_size, device=device)
    
    # 测试GEMM SMID收集
    print("\n1. 测试GEMM SMID收集:")
    grid_dim_x = 10
    grid_dim_y = 1
    
    print(f"运行GEMM，grid=[{grid_dim_x}, {grid_dim_y}]")
    smid_tensor = gemm_custom_grid(A, B, C, grid_dim_x, grid_dim_y)
    
    print(f"返回的SMID张量形状: {smid_tensor.shape}")
    print(f"SMID张量类型: {smid_tensor.dtype}")
    
    if smid_tensor is not None:
        smid_numpy = smid_tensor.cpu().numpy()
        print(f"SMID值: {smid_numpy}")
        
        # 检查有效值
        valid_smids = smid_numpy[smid_numpy >= 0]
        print(f"有效SMID值数量: {len(valid_smids)}/{len(smid_numpy)}")
        if len(valid_smids) > 0:
            unique_smids = np.unique(valid_smids)
            print(f"唯一的SMID值: {unique_smids}")
            print(f"使用的SM数量: {len(unique_smids)}")
    
    # 测试向量加法SMID收集
    print("\n2. 测试向量加法SMID收集:")
    grid_dim_x = 10
    
    print(f"运行向量加法，grid={grid_dim_x}")
    smid_tensor = vector_add(X, Y, Z, grid_dim_x)
    
    print(f"返回的SMID张量形状: {smid_tensor.shape}")
    print(f"SMID张量类型: {smid_tensor.dtype}")
    
    if smid_tensor is not None:
        smid_numpy = smid_tensor.cpu().numpy()
        print(f"SMID值: {smid_numpy}")
        
        # 检查有效值
        valid_smids = smid_numpy[smid_numpy >= 0]
        print(f"有效SMID值数量: {len(valid_smids)}/{len(smid_numpy)}")
        if len(valid_smids) > 0:
            unique_smids = np.unique(valid_smids)
            print(f"唯一的SMID值: {unique_smids}")
            print(f"使用的SM数量: {len(unique_smids)}")
    
    print("\n===== 测试完成 =====")

if __name__ == "__main__":
    test_smid_collection()
