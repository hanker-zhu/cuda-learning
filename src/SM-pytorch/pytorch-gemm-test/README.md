# CUDA内核SM分配与重叠测试框架

这个项目提供了一个测试框架，用于分析CUDA内核在执行时的SM（流多处理器）分配和重叠情况。通过这个工具，您可以：

1. 观察GEMM和向量加法内核使用的SM分布
2. 分析并行执行时SM的重叠使用情况
3. 评估不同SM分配策略对性能的影响
4. 可视化SM分配和重叠情况

## 安装与设置

首先需要构建CUDA扩展模块：

```bash
python setup.py install
```

## 使用方法

### 1. 运行单次测试

测试特定数量的线程块如何影响SM分配：

```bash
./run_tests.sh single 30 20 both
```

参数说明：
- `30`: GEMM内核使用30个线程块
- `20`: 向量加法内核使用20个线程块
- `both`: 运行串行和并行两种模式（可选：`serial`、`parallel`、`both`）

或者直接使用Python脚本：

```bash
python test_kernel_smid.py single --gemm-blocks 30 --vec-blocks 20 --mode both
```

### 2. 运行扫描测试

自动测试不同线程块数量组合的性能和SM重叠情况：

```bash
./run_tests.sh sweep
```

这将生成一个热图，显示不同线程块数量组合下的性能和SM重叠情况。

### 3. 运行预定义的测试组合

```bash
./run_tests.sh combo
```

这将运行一系列预定义的测试组合，生成多个分析结果。

## 输出结果

### 性能数据

脚本将输出以下性能指标：
- 串行执行时间（先GEMM后向量加法）
- 并行执行时间（GEMM和向量加法同时运行）
- 加速比（串行时间/并行时间）

### SMID分析

脚本将分析和显示：
- 每个内核使用的SM ID
- 两个内核重叠使用的SM数量和ID
- SM分配的可视化图表

### 可视化图表

测试会生成以下图表：
- `smid_distribution_serial.png`: 串行执行时的SM分布
- `smid_distribution_parallel.png`: 并行执行时的SM分布
- `sm_allocation_summary.png`: 扫描测试的性能和重叠结果热图

## 实例分析

当您观察到并行执行比串行执行慢时，这通常表明SM资源争用：
- 重叠SM数量高 + 性能下降：表明内核之间存在资源冲突
- 重叠SM数量低 + 性能提升：表明内核能有效地并行执行

## 高级用法

可以修改CUDA内核（`csrc/gemm_kernel.cu`）来测试不同的内核实现对SM分配的影响。例如：
- 调整线程块大小
- 修改内存访问模式
- 改变计算密度

## 故障排除

1. 如果出现"无法导入gemm_test_ext"错误，请确保已运行`python setup.py install`
2. 对于CUDA相关错误，检查CUDA版本兼容性和设备可见性