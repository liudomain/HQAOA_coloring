# HQAOA 图着色算法 - 天衍平台版本

## 项目简介

本项目实现了多层次 QAOA (Multilevel QAOA) 图着色算法，适配天衍量子计算平台，支持大规模图着色问题的求解。项目提供多种子图划分策略，可根据图的规模和稀疏程度选择合适的算法。

## 目录结构

```
QAOA/
├── config_template.py                    # 配置文件模板（需复制为 config.py）
├── graph_loader.py                       # 图数据加载模块
├── multilevel_common_tianyan.py          # 公共函数库（图划分、可视化、冲突计算等）
├── multilevel_standard_QAOA_tianyan.py   # 标准 QAOA 实现（天衍平台适配）
├── Main_Multilevel_qaoa_tianyan.py       # 主程序 - 朴素划分版本（含主函数）
├── Main_Multilevel_qaoa_tianyan_smart.py # 主程序 - 智能划分版本（主函数已移除）
├── Main_Multilevel_qaoa_tianyan_large_naive.py  # 主程序 - 大规模朴素版本（主函数已移除）
├── train_params_tianyan.py               # 参数训练模块
└── README.md                             # 项目说明文档
```

### 输出目录（自动生成）

```
QAOA/
├── logs/                                 # 运行日志
│   ├── standard_qaoa.log
│   ├── standard_smart_qaoa.log
│   ├── large_standard_naive_qaoa.log
│   ├── standard_graph_results.log
│   ├── standard_smart_graph_results.log
│   └── large_standard_naive_graph_results.log
├── csvs/                                 # CSV 结果文件
│   ├── standard_subgraph_results.csv
│   ├── standard_smart_subgraph_results.csv
│   ├── large_standard_naive_subgraph_results.csv
│   ├── standard_graph_results.csv
│   ├── standard_smart_graph_results.csv
│   └── large_standard_naive_graph_results.csv
├── graph_visualizations/                # 图可视化结果
├── subgraph_visualizations/              # 子图可视化结果
└── large_graph_visualizations/           # 大规模图可视化结果
    └── large_subgraph_visualizations/    # 大规模子图可视化结果
```

*注意：所有输出文件和目录已被 `.gitignore` 忽略，不会上传到 GitHub*

## 快速开始

### 1. 安装依赖

```bash
pip install networkx matplotlib numpy pandas cqlib metis
```

### 2. 配置天衍平台

```bash
# 复制配置模板
cd HAdaQAOA/HadaQAOA/adapt_QAOA_Tianyan/QAOA
cp config_template.py config.py

# 编辑 config.py，填入你的天衍平台登录密钥
```

或使用环境变量：

```bash
export TIANYAN_LOGIN_KEY="your_login_key_here"
```

### 3. 运行示例

#### 标准版本（朴素划分 - 推荐）

```bash
# 使用 col 格式数据
python Main_Multilevel_qaoa_tianyan.py --format-type col

# 使用 pkl 格式数据
python Main_Multilevel_qaoa_tianyan.py --format-type pkl

# 自动选择格式（优先 pkl，其次 col）
python Main_Multilevel_qaoa_tianyan.py --format-type auto

# 启用参数训练
python Main_Multilevel_qaoa_tianyan.py --format-type col --train-params

# 指定自定义数据目录
python Main_Multilevel_qaoa_tianyan.py --graph-dir "/path/to/graphs" --format-type pkl
```

#### 智能划分版本

智能划分版本会根据量子比特约束自动调整子图大小，避免退化为贪心着色。

**注意**：此版本的 `main()` 函数已移除，需要通过导入 `main_standard()` 函数来使用。

```python
from Main_Multilevel_qaoa_tianyan_smart import main_standard

# 调用方式与标准版本相同
results = main_standard(
    graphs=graphs,
    dataset="dataset_name",
    graph_index=0,
    seed=10,
    platform=platform,
    lab_id=lab_id,
    trained_params=None,
    train_params=True,
    train_max_iter=20,
    train_lr=0.05
)
```

#### 大规模朴素划分版本

适用于大规模图（如 cora, citeseer），支持可配置的量子比特数限制。

**注意**：此版本的 `main()` 函数已移除，需要通过导入 `main_standard()` 函数来使用。

```python
from Main_Multilevel_qaoa_tianyan_large_naive import main_standard

# 支持更大的量子比特限制
results = main_standard(
    graphs=graphs,
    dataset="large_dataset",
    graph_index=0,
    seed=10,
    platform=platform,
    lab_id=lab_id,
    trained_params=None,
    train_params=True,
    train_max_iter=100,
    train_lr=0.01,
    max_qubits=200  # 默认 200，可根据机时包限制调整
)
```

## 命令行参数

### 通用参数（Main_Multilevel_qaoa_tianyan.py）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--graph-dir` | str | None | 图数据目录路径（指定后从此目录加载） |
| `--format-type` | str | auto | 数据加载格式：auto(自动)/col(.col only)/pkl(.pkl only) |
| `--seed` | int | 10 | 随机种子 |
| `--train-params` | flag | False | 是否训练参数 |
| `--train-max-iter` | int | 20 | 参数训练最大迭代次数 |
| `--train-lr` | float | 0.05 | 参数训练学习率 |
| `--train-epsilon-fd` | float | 0.05 | 有限差分步长 |
| `--enable-cache` | flag | False | 启用参数缓存 |
| `--dataset` | str | test_dataset | 数据集名称 |

### 大规模版本专属参数（Main_Multilevel_qaoa_tianyan_large_naive.py）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--max-qubits` | int | 200 | 量子比特数限制（根据机时包限制调整） |

### 完整参数示例

```bash
# 标准版本完整参数
python Main_Multilevel_qaoa_tianyan.py \
    --format-type col \
    --seed 42 \
    --train-params \
    --train-max-iter 50 \
    --train-lr 0.01 \
    --dataset my_dataset \
    --graph-dir /path/to/graphs

# 大规模版本调整量子比特限制
python Main_Multilevel_qaoa_tianyan_large_naive.py \
    --max-qubits 32 \
    --train-params \
    --train-max-iter 100 \
    --train-lr 0.01
```

## 三种版本对比

| 特性 | 朴素划分 | 智能划分 | 大规模朴素 |
|------|---------|---------|-----------|
| **主程序文件** | Main_Multilevel_qaoa_tianyan.py | Main_Multilevel_qaoa_tianyan_smart.py | Main_Multilevel_qaoa_tianyan_large_naive.py |
| **主函数状态** | ✅ 保留 | ❌ 移除（需调用 main_standard） | ❌ 移除（需调用 main_standard） |
| **子图划分策略** | divide_graph（固定数量） | smart_divide_graph_with_qubit_constraint（自适应） | divide_graph（固定数量） |
| **量子比特限制** | 36（固定） | 36（固定） | 可配置（默认 200） |
| **适用场景** | 小规模图，标准 QAOA | 中等规模图，需避免贪心退化 | 大规模图，支持更多量子比特 |
| **日志文件** | standard_qaoa.log | standard_smart_qaoa.log | large_standard_naive_qaoa.log |
| **子图结果 CSV** | standard_subgraph_results.csv | standard_smart_subgraph_results.csv | large_standard_naive_subgraph_results.csv |
| **图结果 CSV** | standard_graph_results.csv | standard_smart_graph_results.csv | large_standard_naive_graph_results.csv |
| **可视化目录** | graph_visualizations/ | graph_visualizations/ | large_graph_visualizations/ |

## 算法说明

### 多层次 QAOA 流程

```
1. 图划分
   ├─ 朴素划分：使用 METIS 进行固定数量的子图划分
   ├─ 智能划分：根据量子比特约束自动调整子图大小
   └─ 稀疏度检测：根据图的稀疏程度优化划分参数

2. 子图着色
   ├─ 孤立节点：直接使用 1 色着色
   ├─ 链式图：使用 2 色专用着色算法
   ├─ 环图：偶环用 2 色，奇环用 3 色
   ├─ 完全图：使用贪心着色（节省量子资源）
   └─ 普通图：使用 QAOA 进行着色（带参数训练）

3. 全局优化
   ├─ 合并子图着色结果
   ├─ 冲突边检测
   └─ 迭代优化边界节点颜色
```

### 天衍平台适配

- **库支持**：使用 `cqlib` 库与天衍平台交互
- **真机执行**：支持 tianyan_sa 和 tianyan_sw 两台量子计算机
- **参数训练**：集成 ADAM 优化器进行 QAOA 参数训练
- **热启动**：逐步增加 k 值时使用上一轮的最优参数
- **早停机制**：找到无冲突解后立即停止训练

### 智能子图划分

`smart_divide_graph_with_qubit_constraint()` 函数实现了智能划分策略：

1. 根据最大颜色数 `max_k` 计算每节点需要的比特数：`bits_per_node = ceil(log2(max_k))`
2. 计算最大允许节点数：`max_nodes = max_qubits // bits_per_node`
3. 初始子图数：`num_subgraphs = ceil(total_nodes / max_nodes)`
4. 使用 METIS 进行初始划分
5. 递归二分超限子图，直到所有子图满足量子比特约束

这种策略避免了大规模图在 QAOA 求解时退化为贪心着色。

## 配置说明

### config.py 配置项

```python
TIANYAN_CONFIG = {
    "login_key": "your_login_key_here",  # 天衍平台登录密钥
    "lab_id": None,                      # 实验室ID（可选，不设置会自动创建）
}

ALGORITHM_PARAMS = {
    "max_k": 20,        # 最大颜色数
    "p": 1,            # QAOA 层数
    "num_shots": 1000, # 量子测量次数
    "max_iter": 20,    # 最大迭代次数
    "early_stop_threshold": 5,  # 早停阈值
    "Q": 20,           # METIS 平衡因子
}
```

### 或使用环境变量

```bash
# Linux/Mac
export TIANYAN_LOGIN_KEY="your_login_key_here"

# Windows (PowerShell)
$env:TIANYAN_LOGIN_KEY="your_login_key_here"

# Windows (CMD)
set TIANYAN_LOGIN_KEY=your_login_key_here
```

## 输出文件说明

### 日志文件

| 文件名 | 说明 |
|--------|------|
| `logs/standard_qaoa.log` | 朴素划分版本运行日志 |
| `logs/standard_smart_qaoa.log` | 智能划分版本运行日志 |
| `logs/large_standard_naive_qaoa.log` | 大规模朴素版本运行日志 |
| `logs/standard_graph_results.log` | 朴素划分图级别结果 |
| `logs/standard_smart_graph_results.log` | 智能划分图级别结果 |
| `logs/large_standard_naive_graph_results.log` | 大规模朴素图级别结果 |

### CSV 文件

**子图级别结果**：

```csv
dataset, graph_name, graph_index, subgraph_index, nodes, edges, min_k, conflicts, status, processing_time
```

**图级别结果**：

```csv
dataset, graph_name, graph_index, nodes, edges, final_conflicts, total_edges, 
final_accuracy, unique_colors, global_max_k, best_k_value, 
subgraph_reoptimization_count, processing_time, conflict_changes, 
total_time, train_params, train_max_iter, train_lr
```

### 可视化图片

| 目录 | 说明 |
|------|------|
| `graph_visualizations/` | 朴素划分和智能划分的图可视化 |
| `subgraph_visualizations/` | 朴素划分和智能划分的子图可视化 |
| `large_graph_visualizations/` | 大规模版本的图可视化 |
| `large_subgraph_visualizations/` | 大规模版本的子图可视化 |

**图片类型**：
- `*_original.png` - 原始图
- `*_subgraphs_renumbered.png` - 子图（新编号，对应量子比特）
- `*_subgraphs_original.png` - 子图（原始节点 ID）
- `*_colored_subgraphs_renumbered.png` - 着色子图（新编号）
- `*_colored_subgraphs_original.png` - 着色子图（原始 ID）
- `*_final_coloring.png` - 最终着色结果

## 常见问题

### Q1: 提示"请在 config.py 中配置 TIANYAN_CONFIG['login_key'] 或设置环境变量"

**A**: 需要创建 `config.py` 文件：

```bash
cp config_template.py config.py
```

然后编辑 `config.py`，填入你的天衍平台登录密钥。或者直接设置环境变量：

```bash
export TIANYAN_LOGIN_KEY="your_login_key_here"
```

### Q2: 提示"最大比特数不支持本任务"

**A**: 这是因为子图所需的量子比特数超过了机时包限制。使用 `--max-qubits` 参数调整量子比特数限制：

```bash
# 标准版本：默认 36 比特
python Main_Multilevel_qaoa_tianyan.py --format-type col

# 大规模版本：默认 200 比特，可降低
python Main_Multilevel_qaoa_tianyan_large_naive.py --max-qubits 32
```

较小的 `max_qubits` 值会导致更小的子图，可能影响着色质量。

### Q3: 如何使用智能划分版本？

**A**: 智能划分版本的主函数已移除，需要通过编程方式调用：

```python
from Main_Multilevel_qaoa_tianyan_smart import main_standard
from graph_loader import load_graphs_from_dir
from cqlib import TianYanPlatform

# 加载图数据
graphs = load_graphs_from_dir('default', format_type='col')

# 初始化天衍平台
platform = TianYanPlatform(login_key="your_login_key")
platform.set_machine("tianyan_sw")

# 创建实验室
lab_id = platform.create_lab(name='my_lab', remark='Test')

# 调用智能划分版本
results = main_standard(
    graphs=graphs,
    dataset="test_dataset",
    graph_index=0,
    seed=10,
    platform=platform,
    lab_id=lab_id,
    train_params=True,
    train_max_iter=20,
    train_lr=0.05
)
```

### Q4: 如何查看生成的图片？

**A**: 图片保存在以下目录中：

- 朴素划分/智能划分：
  - `graph_visualizations/` - 图可视化
  - `subgraph_visualizations/` - 子图可视化
- 大规模版本：
  - `large_graph_visualizations/` - 图可视化
  - `large_subgraph_visualizations/` - 子图可视化

图片格式为 PNG，可直接用图片查看器打开。

### Q5: 如何禁用图片生成？

**A**: 目前代码中图片生成是硬编码的，如需禁用，可以在代码中注释掉以下函数调用：

```python
# 在 main_standard() 函数中注释掉这些行
# plot_original_graph(...)
# plot_New_IDs_subgraphs(...)
# plot_Original_IDs_subgraphs(...)
# plot_New_IDs_colored_subgraphs(...)
# plot_Original_IDs_colored_subgraphs(...)
# visualize_graph(...)
```

或者依赖 `.gitignore` 的配置，生成的图片不会被上传到 GitHub。

### Q6: 训练参数的含义是什么？

**A**:

- `--train-max-iter`: 参数训练的最大迭代次数，默认 20。增加此值可能获得更好的参数，但训练时间更长。
- `--train-lr`: 参数训练的学习率，默认 0.05。较大的学习率收敛更快，但可能震荡。
- `--train-epsilon-fd`: 有限差分步长，用于计算梯度，默认 0.05。

### Q7: 支持哪些图数据格式？

**A**:

- `.col` 格式：标准图着色数据集格式
- `.pkl` 格式：Python pickle 序列化的 NetworkX 图对象
- `auto`：自动选择格式，优先使用 `.pkl`，其次使用 `.col`

### Q8: 如何处理超大规模图？

**A**: 对于超大规模图（如 > 10000 节点）：

1. 使用大规模版本：`Main_Multilevel_qaoa_tianyan_large_naive.py`
2. 根据机时包限制调整 `--max-qubits` 参数
3. 增加 `--train-max-iter` 以获得更好的参数
4. 考虑降低 `--train-lr` 以提高稳定性

### Q9: 算法输出的准确率如何计算？

**A**:

```
准确率 = 1 - (冲突边数 / 总边数)
```

例如：总边数 100，冲突边数 5，则准确率为 95%。

### Q10: 如何复现实验结果？

**A**: 使用固定的随机种子：

```bash
python Main_Multilevel_qaoa_tianyan.py --seed 42 --format-type col
```

相同的随机种子会生成相同的图划分和初始着色，但由于量子测量的随机性，最终的着色结果可能略有不同。

## 核心函数说明

### multilevel_common_tianyan.py

提供共享的工具函数：

- **图处理**：
  - `divide_graph()` - 朴素子图划分（使用 METIS）
  - `smart_divide_graph_with_qubit_constraint()` - 智能子图划分
  - `detect_sparse_graph_strategy()` - 检测图的稀疏程度
  - `calculate_optimal_subgraph_params()` - 计算最优子图参数

- **图类型判断**：
  - `is_complete_graph()` - 判断是否为完全图
  - `is_cycle_graph()` - 判断是否为环图
  - `is_chain_graph()` - 判断是否为链式图

- **着色算法**：
  - `count_conflicts()` - 计算冲突数
  - `chain_graph_coloring()` - 链式图着色
  - `cycle_graph_coloring()` - 环图着色
  - `_greedy_coloring_from_max_degree()` - 贪心着色
  - `_greedy_coloring_optimal_k()` - 优化 k 值的贪心着色

- **可视化**：
  - `plot_original_graph()` - 原始图可视化
  - `plot_New_IDs_subgraphs()` - 子图可视化（新编号）
  - `plot_Original_IDs_subgraphs()` - 子图可视化（原始 ID）
  - `plot_New_IDs_colored_subgraphs()` - 着色子图可视化（新编号）
  - `plot_Original_IDs_colored_subgraphs()` - 着色子图可视化（原始 ID）
  - `visualize_graph()` - 最终着色可视化

- **QAOA 核心**：
  - `qaoa_ansatz_tianyan()` - 构建 QAOA 线路
  - `solve_k_coloring_tianyan_with_training()` - 训练并执行 QAOA
  - `extract_coloring_tianyan()` - 从测量结果提取着色

### multilevel_standard_QAOA_tianyan.py

提供标准 QAOA 的多层次实现：

- `sequential_process_subgraphs_tianyan()` - 顺序处理子图着色
- `iterative_optimization_tianyan()` - 迭代优化全局着色
- `plot_energy_convergence()` - 绘制能量收敛曲线

### train_params_tianyan.py

提供参数训练功能：

- `QAOAParamOptimizer` - QAOA 参数优化器类
- `optimize_adam_pytorch()` - 使用 ADAM 优化器训练参数

## 参考资料

- [天衍平台文档](https://quantum.tencent.com/)
- [QAOA 算法论文](https://arxiv.org/abs/1411.4028)
- [cqlib 文档](https://github.com/Tencent-Quantum-Lab/cqlib)
- 项目根目录的 `GITHUB_SECURITY_NOTES.md` 包含安全配置详细说明

## 贡献指南

欢迎提交 Issue 和 Pull Request！

提交代码前请确保：

1. 代码符合 PEP 8 规范
2. 添加必要的注释和文档字符串
3. 确保敏感信息（如 login_key）不会被提交
4. 不要提交生成的日志、CSV 和图片文件

## 许可证

本项目采用 MIT 许可证。详情请参阅 LICENSE 文件。
