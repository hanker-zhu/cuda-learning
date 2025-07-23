import torch
import numpy as np
import sys

def debug_smid_collection():
    """专门用于调试SMID收集的函数"""
    try:
        # 导入扩展
        from gemm_test_ext._C import gemm_custom_grid, vector_add
        print("成功导入CUDA扩展")
        
        # 准备小型测试数据
        device = 'cuda:0'
        M, N, K = 64, 64, 64
        A = torch.randn(M, K, device=device)
        B = torch.randn(K, N, device=device)
        C = torch.zeros(M, N, device=device)
        
        vec_size = 1024
        X = torch.randn(vec_size, device=device)
        Y = torch.randn(vec_size, device=device)
        Z = torch.zeros(vec_size, device=device)
        
        print("\n==== 测试GEMM SMID收集 ====")
        gemm_blocks_x = 4
        gemm_blocks_y = 4
        total_blocks = gemm_blocks_x * gemm_blocks_y
        print(f"GEMM Grid大小: {gemm_blocks_x}x{gemm_blocks_y} = {total_blocks}个线程块")
        
        # 运行GEMM并检查返回值
        smids = gemm_custom_grid(A, B, C, gemm_blocks_x, gemm_blocks_y)
        
        if smids is None:
            print("错误: gemm_custom_grid返回了None")
            return False
            
        if not torch.is_tensor(smids):
            print(f"错误: gemm_custom_grid返回了非张量对象: {type(smids)}")
            return False
            
        print(f"返回的SMID张量形状: {smids.shape}")
        print(f"SMID张量内容: {smids}")
        
        # 分析SMID分布
        smids_np = smids.cpu().numpy()
        
        # 检查是否有有效的SMID (大于0)
        valid_smids = smids_np[smids_np > 0]
        if len(valid_smids) == 0:
            print("警告: 没有检测到有效的SMID值 (所有值为0)")
        else:
            unique_smids = np.unique(valid_smids)
            print(f"检测到的唯一SMID: {unique_smids}")
            print(f"检测到的SM数量: {len(unique_smids)}")
            
            # 每个SM上有多少线程块
            for smid in unique_smids:
                count = np.sum(smids_np == smid)
                print(f"SM {smid}: {count}个线程块")
        
        print("\n==== 测试向量加法SMID收集 ====")
        vec_blocks = 8
        print(f"向量加法Grid大小: {vec_blocks}个线程块")
        
        # 运行向量加法并检查返回值
        smids = vector_add(X, Y, Z, vec_blocks)
        
        if smids is None:
            print("错误: vector_add返回了None")
            return False
            
        if not torch.is_tensor(smids):
            print(f"错误: vector_add返回了非张量对象: {type(smids)}")
            return False
            
        print(f"返回的SMID张量形状: {smids.shape}")
        print(f"SMID张量内容: {smids}")
        
        # 分析SMID分布
        smids_np = smids.cpu().numpy()
        
        # 检查是否有有效的SMID (大于0)
        valid_smids = smids_np[smids_np > 0]
        if len(valid_smids) == 0:
            print("警告: 没有检测到有效的SMID值 (所有值为0)")
        else:
            unique_smids = np.unique(valid_smids)
            print(f"检测到的唯一SMID: {unique_smids}")
            print(f"检测到的SM数量: {len(unique_smids)}")
            
            # 每个SM上有多少线程块
            for smid in unique_smids:
                count = np.sum(smids_np == smid)
                print(f"SM {smid}: {count}个线程块")
        
        print("\n调试完成")
        return True
    
    except Exception as e:
        print(f"调试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始SMID收集调试...")
    success = debug_smid_collection()
    sys.exit(0 if success else 1)
