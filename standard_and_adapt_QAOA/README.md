# 标准与自适应 QAOA 图着色

本目录包含多层次 QAOA 图着色算法的 MindQuantum 实现，支持标准 QAOA、自适应 QAOA 和含噪声 QAOA。

## 项目简介

基于 MindQuantum 框架实现的多层次量子近似优化算法（HQAOA），用于解决图着色问题。支持通过仿真器进行算法实验和性能对比。

**核心特性**:
- 三种 QAOA 变体：标准 QAOA、自适应 QAOA、含噪声 QAOA
- 多层次划分策略：朴素划分、智能划分
- 大规模图处理：支持数千节点的图
- 完整的可视化和日志系统
- 与经典算法对比（贪心、Welch-Powell）

## 目录结构

```
standard_and_adapt_QAOA/
├── multilevel_common.py                      # 公共函数库（图划分、可视化、QAOA核心）
├── multilevel_adapt_QAOA_k_coloring.py       # 自适应 QAOA 实现
├── multilevel_standard_QAOA_k_coloring.py   # 标准 QAOA 实现
├── multilevel_adapt_noise_QAOA_k_coloring.py # 含噪声 QAOA 实现
├── graph_loader.py                           # 图数据加载模块
├── unified_data_processor.py                # 统一数据处理
├── Main_Multilevel_qaoa.py                   # 主程序（朴素划分版本，含 main 函数）
├── Main_Multilevel_qaoa_smart_divide.py      # 主程序（智能划分版本）
├── Main_Multilevel_qaoa_large_graph.py       # 主程序（大规模图版本）
├── run_experiments.py                        # 实验运行脚本
└── run_experiments.ipynb                     # Jupyter 实验笔记本（已被 .gitignore 忽略）
```

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install mindspore>=2.0 mindquantum>=0.9 networkx>=2.8 matplotlib>=3.5 numpy>=1.21 pandas>=1.3 metis>=0.2.0
```

### 2. 运行实验

#### 使用主程序（朴素划分）

```bash
# 运行自适应 QAOA（使用朴素划分）
python Main_Multilevel_qaoa.py

# 通过编程方式调用
from Main_Multilevel_qaoa import main_adapt, main_standard, main_adapt_noise

# 加载图数据
from graph_loader import load_graphs_from_dir
graphs = load_graphs_from_dir(format_type='col')

# 运行算法
results = main_adapt(graphs, dataset="test", graph_index=0, seed=10)
```

#### 使用主程序（智能划分）

```bash
# 智能划分版本需要通过编程方式调用
python -c "
from Main_Multilevel_qaoa_smart_divide import main_adapt as main_adapt_smart
from graph_loader import load_graphs_from_dir
graphs = load_graphs_from_dir(format_type='col')
main_adapt_smart(graphs, dataset='test', graph_index=0, seed=10)
"
```

#### 使用实验脚本

```bash
# 运行自适应 QAOA
python run_experiments.py --adapt

# 运行标准 QAOA
python run_experiments.py --standard

# 运行含噪声 QAOA
python run_experiments.py --adapt-noise --noise-prob 0.05

# 运行所有算法对比
python run_experiments.py --adapt --standard --adapt-noise

# 使用自定义图目录和格式
python run_experiments.py --adapt --graph-dir ./data --format-type col
```

#### 处理大规模图

```bash
# 处理大规模图（citeseer、pubmed 等）
python Main_Multilevel_qaoa_large_graph.py
```

## 算法说明

### 1. 标准 QAOA (Standard QAOA)

**文件**: `multilevel_standard_QAOA_k_coloring.py`

**核心函数**: `solve_k_coloring_standard()`

**特点**:
- 使用固定的 X 门作为混合算子
- 每层的混合算子相同
- 参数数量较少，优化相对简单
- 适合基准测试和性能对比

**主程序**: `Main_Multilevel_qaoa.py` 中的 `main_standard()`

### 2. 自适应 QAOA (Adaptive QAOA)

**文件**: `multilevel_adapt_QAOA_k_coloring.py`

**核心函数**: `solve_k_coloring()`

**特点**:
- 梯度选择最优混合算子
- 每层可能使用不同的混合算子
- 可能有更好的解质量，但计算开销更大
- 适合需要高性能的场景

**主程序**: `Main_Multilevel_qaoa.py` 中的 `main_adapt()`

### 3. 含噪声 QAOA (Adaptive QAOA with Noise)

**文件**: `multilevel_adapt_noise_QAOA_k_coloring.py`

**核心函数**: `solve_k_coloring_noise()`

**特点**:
- 模拟退极化噪声（Depolarizing Noise）
- 研究噪声对算法的影响
- 支持自定义噪声概率
- 适合真实量子设备模拟

**主程序**: `Main_Multilevel_qaoa.py` 中的 `main_adapt_noise()`

**噪声参数**:
```python
noise_params = {
    "noise_prob": 0.05,           # 退极化噪声概率
    "noise_type": "depolarizing"  # 噪声类型
}
```

### 4. 三种主程序对比

| 特性 | Main_Multilevel_qaoa.py | Main_Multilevel_qaoa_smart_divide.py | Main_Multilevel_qaoa_large_graph.py |
|------|-------------------------|--------------------------------------|--------------------------------------|
| 子图划分策略 | `divide_graph` (朴素) | `smart_divide_graph_with_qubit_constraint` (智能) | `smart_divide_graph_with_qubit_constraint` |
| 主函数 | ✅ 完整 main() | ❌ 无 main() | ❌ 无 main() |
| 量子比特限制 | 36 | 36 | 200（可配置） |
| 适用场景 | 中小规模图 | 中小规模图（需避免贪心退化） | 大规模图 |
| 日志文件 | `logs/adapt_graph_results.log` | `logs/smart_adapt_graph_results.log` | `logs/large_adapt_graph_results.log` |
| CSV 文件 | `csvs/adapt_subgraph_results.csv` | `csvs/smart_adapt_subgraph_results.csv` | `csvs/large_adapt_subgraph_results.csv` |

## 算法流程

### 多层次 QAOA 流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     1. 图划分（Divide）                        │
│  - 朴素划分: divide_graph(num_subgraphs, max_nodes)          │
│  - 智能划分: smart_divide_graph_with_qubit_constraint()      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  2. 子图 QAOA 着色（Color）                     │
│  - 孤立节点: 批量处理，所有节点同色                            │
│  - 环图: 专用算法（偶环2色，奇环3色）                         │
│  - 完全图: 贪心着色                                           │
│  - 普通图: QAOA + 贪心混合策略                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                3. 合并子图结果（Merge）                         │
│  - 统一子图颜色编号，避免冲突                                 │
│  - 处理边界节点的颜色冲突                                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              4. 边界节点优化（Boundary Optimization）           │
│  - 检测边界冲突                                               │
│  - 贪心策略重新着色边界节点                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              5. 迭代改进（Iterative Optimization）             │
│  - 多次重新着色冲突子图                                       │
│  - 检测同构子图并复用结果                                     │
│  - 渐进式优化颜色数                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 子图划分策略

#### 朴素划分 (`divide_graph`)

```python
subgraphs, mappings = divide_graph(
    graph,
    num_subgraphs=4,    # 目标子图数量
    Q=20,              # METIS 平衡因子
    max_nodes=10       # 每个子图最大节点数
)
```

**特点**:
- 使用 METIS 库进行图划分
- 对大子图递归二分直至节点数 ≤ max_nodes
- 简单快速，但可能退化为贪心着色

#### 智能划分 (`smart_divide_graph_with_qubit_constraint`)

```python
subgraphs, mappings = smart_divide_graph_with_qubit_constraint(
    graph,
    max_qubits=36,     # 量子比特限制
    n_qubits_per_node=2 # 每个节点编码所需量子比特
)
```

**特点**:
- 根据量子比特约束自动调整子图大小
- 避免退化为贪心着色
- 递归二分超限子图
- 适合实际量子设备限制

## 数据格式

### 支持的数据格式

#### 1. DIMACS COLOR 格式 (.col)

```
c Comment line
p edge <num_nodes> <num_edges>
e <node1> <node2>
e <node1> <node2>
...
```

**示例**:
```
c Queen 6x6 graph
p edge 36 90
e 1 2
e 1 7
e 2 3
...
```

#### 2. Pickle 序列化格式 (.pkl)

- NetworkX Graph 对象序列化
- 支持任意 Python 对象

### 数据加载策略

```python
from graph_loader import load_graphs_from_dir

# 自动检测格式（优先 .col，若无则加载 .pkl）
graphs = load_graphs_from_dir('default', format_type='auto')

# 只加载 .col 文件
graphs = load_graphs_from_dir('./data', format_type='col')

# 只加载 .pkl 文件
graphs = load_graphs_from_dir('./data', format_type='pkl')
```

## 参数配置

### 算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_qubits_per_node` | 2 | 每个节点编码所需量子比特数 |
| `learning_rate` | 0.01 | 优化器学习率 |
| `max_k` | 20 | 最大颜色数 |
| `p` | 3 | QAOA 层数（电路深度） |
| `num_steps` | 1000 | 最大训练步数 |
| `max_iter` | 10 | 迭代优化次数 |
| `adjacency_threshold` | 0.3 | 邻接矩阵阈值（判断图类型） |
| `early_stop_threshold` | 3 | 连续相同训练值的提前退出阈值 |
| `penalty` | 1000 | 冲突惩罚系数 |
| `Q` | 20 | 图划分平衡因子 |

### 噪声参数（仅含噪声 QAOA）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `noise_prob` | 0.05 | 退极化噪声概率 |
| `noise_type` | "depolarizing" | 噪声类型 |

### 可视化配置

```python
visualization_config = {
    "save_plots": True,          # 保存图表
    "show_plots": False,         # 显示图表（非交互模式）
    "plot_format": "pdf",         # 图表格式
    "dpi": 800                    # 分辨率
}
```

## 输出说明

### 目录结构

```
HAdaQAOA/
├── logs/                                    # 日志文件
│   ├── adapt_graph_results.log              # 自适应 QAOA 日志
│   ├── standard_graph_results.log           # 标准 QAOA 日志
│   ├── adapt_noise_graph_results.log       # 含噪声 QAOA 日志
│   └── *.log                                # 其他日志
├── csvs/                                    # 结果文件
│   ├── adapt_subgraph_results.csv          # 自适应 QAOA 子图结果
│   ├── standard_subgraph_results.csv       # 标准 QAOA 子图结果
│   ├── all_results.csv                     # 实验对比结果
│   └── large_datasets_classical_results.csv # 大规模图结果
├── graph_visualizations/                   # 原始图可视化
│   └── *.pdf
├── subgraph_visualizations/                # 子图可视化
│   └── *.pdf
└── experiment_visualizations/              # 实验对比图表
    └── *.pdf
```

*注意: 输出文件已被 `.gitignore` 忽略，不会上传到 GitHub*

### 日志文件格式

#### 子图结果 CSV (`adapt_subgraph_results.csv`)

```csv
dataset,graph_name,graph_index,subgraph_index,nodes,edges,min_k,conflicts,status,processing_time
test_dataset,queen6_6.col,0,0,10,15,4,0,success,1.234
test_dataset,queen6_6.col,0,1,10,15,3,0,success,0.987
...
```

#### 图结果 CSV (`adapt_graph_results.csv`)

```csv
dataset,graph_name,graph_index,nodes,edges,final_conflicts,total_edges,final_accuracy,unique_colors,global_max_k,best_k_value,subgraph_reoptimization_count,processing_time,conflict_changes,total_time
test_dataset,queen6_6.col,0,36,90,0,90,1.000,6,20,6,3,15.678,0,15.678
...
```

#### 实验对比 CSV (`all_results.csv`)

```csv
graph_file,graph_index,adapt_colors,adapt_time,adapt_success,std_colors,std_time,std_success,adapt_colors_noise,adapt_time_noise,adapt_success_noise,noise_prob_used
queen6_6.col,0,6,12.34,True,7,10.56,True,7,14.78,True,0.05
...
```

### 可视化文件

- **原始图可视化**: `graph_visualizations/{graph_name}_original.pdf`
- **子图可视化**: `subgraph_visualizations/{graph_name}_subgraphs.pdf`
- **着色结果**: `subgraph_visualizations/{graph_name}_colored.pdf`
- **实验对比**: `experiment_visualizations/{seed}_performance_dashboard.pdf`

## 使用示例

### 基础使用

```python
from graph_loader import load_graphs_from_dir
from Main_Multilevel_qaoa import main_adapt, main_standard, main_adapt_noise

# 加载图数据
graphs = load_graphs_from_dir(format_type='auto')

# 运行自适应 QAOA
results = main_adapt(
    graphs=graphs,
    dataset="test_dataset",
    graph_index=0,
    seed=10
)

# 运行标准 QAOA
results = main_standard(
    graphs=graphs,
    dataset="test_dataset",
    graph_index=0,
    seed=10
)

# 运行含噪声 QAOA
results = main_adapt_noise(
    graphs=graphs,
    dataset="test_dataset",
    graph_index=0,
    seed=10,
    depolarizing_prob=0.05
)
```

### 自定义参数

```python
results = main_adapt(
    graphs=graphs,
    dataset="test_dataset",
    graph_index=0,
    seed=10,
    max_k=15,      # 自定义最大颜色数
    p=2,           # 使用 2 层 QAOA
    max_iter=30    # 增加迭代次数
)
```

### 智能划分版本

```python
from Main_Multilevel_qaoa_smart_divide import main_adapt as main_adapt_smart

# 使用智能划分（避免贪心退化）
results = main_adapt_smart(
    graphs=graphs,
    dataset="test_dataset",
    graph_index=0,
    seed=10
)
```

### 大规模图处理

```bash
# 处理大规模图（citeseer、pubmed 等）
python Main_Multilevel_qaoa_large_graph.py
```

**支持的数据集**:
- `cora.col`
- `citeseer.col`
- `pubmed.col`

**特点**:
- 量子比特限制 200
- 经典算法对比（Greedy、Welch-Powell）
- 验证图的可处理性
- 详细的统计摘要

## 核心函数说明

### multilevel_common.py

#### 图处理函数

| 函数 | 说明 |
|------|------|
| `divide_graph(graph, num_subgraphs, Q, max_nodes)` | 朴素划分图 |
| `smart_divide_graph_with_qubit_constraint(graph, max_qubits, n_qubits_per_node)` | 智能划分图 |
| `is_complete_graph(graph)` | 判断是否为完全图 |
| `is_cycle_graph(graph)` | 判断是否为环图 |
| `is_odd_cycle(graph)` | 判断是否为奇环 |
| `get_graph_signature(graph)` | 获取图签名（用于缓存） |

#### QAOA 核心函数

| 函数 | 说明 |
|------|------|
| `build_hamiltonian(graph, k, penalty)` | 构建 QAOA 哈密顿量 |
| `adapt_qaoa_ansatz(graph, k, p, params)` | 构建自适应 QAOA 线路 |
| `qaoa_ansatz_standard(graph, k, p, params)` | 构建标准 QAOA 线路 |
| `qaoa_cost(params, circ, sim, H)` | QAOA 成本函数 |
| `qaoa_mixer(params, circ, sim, H)` | QAOA 混合算子 |
| `derivative(params, circ, sim, H, param_idx, mixer_op)` | 计算梯度 |

#### 着色辅助函数

| 函数 | 说明 |
|------|------|
| `count_conflicts(coloring, graph)` | 计算着色冲突数 |
| `extract_coloring(bitstring, k, n_qubits_per_node)` | 从测量结果提取着色 |
| `cycle_graph_coloring(graph)` | 环图专用着色 |
| `_greedy_coloring_from_max_degree(graph)` | 贪心着色（度数降序） |
| `_resolve_conflicts_with_greedy(graph, coloring, k)` | 贪心解决冲突 |

#### 可视化函数

| 函数 | 说明 |
|------|------|
| `plot_original_graph(graph, filename)` | 可视化原始图 |
| `plot_New_IDs_subgraphs(subgraphs, filename)` | 可视化子图（新ID） |
| `plot_Original_IDs_subgraphs(subgraphs, mappings, filename)` | 可视化子图（原始ID） |
| `plot_New_IDs_colored_subgraphs(subgraphs, colorings, filename)` | 可视化着色子图（新ID） |
| `plot_Original_IDs_colored_subgraphs(...)` | 可视化着色子图（原始ID） |
| `visualize_graph(graph, coloring, filename, ...)` | 可视化着色结果 |

### multilevel_adapt_QAOA_k_coloring.py

| 函数 | 说明 |
|------|------|
| `solve_k_coloring(graph, k, p, ...)` | 自适应 QAOA 求解 k 着色 |
| `sequential_process_subgraphs(...)` | 顺序处理子图着色 |
| `iterative_optimization(...)` | 迭代优化全局着色 |

### multilevel_standard_QAOA_k_coloring.py

| 函数 | 说明 |
|------|------|
| `solve_k_coloring_standard(graph, k, p, ...)` | 标准 QAOA 求解 k 着色 |
| `sequential_process_subgraphs_standard(...)` | 顺序处理子图着色 |
| `iterative_optimization_standard(...)` | 迭代优化全局着色 |

### multilevel_adapt_noise_QAOA_k_coloring.py

| 函数 | 说明 |
|------|------|
| `solve_k_coloring_noise(graph, k, p, ...)` | 含噪声自适应 QAOA 求解 k 着色 |
| `sequential_process_subgraphs_noise(...)` | 顺序处理子图着色（含噪声） |
| `iterative_optimization_noise(...)` | 迭代优化全局着色（含噪声） |

## 命令行参数

### run_experiments.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--adapt` | flag | False | 运行 Adapt-QAOA |
| `--standard` | flag | False | 运行 Standard-QAOA |
| `--adapt-noise` | flag | False | 运行含噪 Adapt-QAOA |
| `--noise-prob` | float | 0.05 | 噪声概率 |
| `--seed` | int | 10 | 随机种子 |
| `--graph-dir` | str | None | 图数据目录路径 |
| `--format-type` | str | auto | 数据格式（auto/col/pkl） |

**示例**:
```bash
# 运行单一算法
python run_experiments.py --adapt
python run_experiments.py --standard
python run_experiments.py --adapt-noise --noise-prob 0.1

# 运行多个算法对比
python run_experiments.py --adapt --standard
python run_experiments.py --adapt --standard --adapt-noise

# 使用自定义图目录和格式
python run_experiments.py --adapt --graph-dir /path/to/graphs --format-type col
```

## 性能对比

### 算法对比

| 算法 | 混合算子 | 参数数量 | 解质量 | 计算开销 | 适用场景 |
|------|----------|----------|--------|----------|----------|
| 标准 QAOA | 固定 X 门 | 少 | 中等 | 低 | 基准测试、快速实验 |
| 自适应 QAOA | 梯度选择 | 多 | 好 | 高 | 需要高质量解 |
| 含噪声 QAOA | 梯度选择 | 多 | 中等 | 高 | 真实设备模拟 |

### 典型结果对比

| 图名称 | 节点数 | Greedy | Welch-Powell | 标准 QAOA | 自适应 QAOA |
|--------|--------|--------|--------------|-----------|-------------|
| queen6_6 | 36 | 8 | 7 | 7 | 6 |
| le450_15c | 450 | 18 | 17 | 16 | 15 |
| school1 | 385 | 20 | 18 | 17 | 15 |

## 常见问题

### Q1: 提示 "MindQuantum not installed"

**A**: 安装 MindQuantum:
```bash
pip install mindquantum
```

### Q2: 如何禁用可视化？

**A**: 修改 `matplotlib.use('Agg')` 已设置为非交互模式，如需完全禁用可视化，可在代码中注释掉 `plot_*` 函数调用。

### Q3: 运行时间过长怎么办？

**A**: 减少以下参数:
- 减小 `max_k` (最大颜色数，如从 20 降到 15)
- 减小 `p` (QAOA 层数，如从 3 降到 1)
- 减小 `num_steps` (最大训练步数，如从 1000 降到 500)
- 减小 `max_iter` (迭代优化次数，如从 10 降到 5)

### Q4: 如何使用自己的图数据？

**A**: 将图数据保存为 `.col` 或 `.pkl` 格式，并使用 `--graph-dir` 参数指定目录:
```bash
python run_experiments.py --adapt --graph-dir /path/to/graphs --format-type col
```

### Q5: 朴素划分和智能划分如何选择？

**A**:
- **朴素划分**: 适合中小规模图，快速但可能退化为贪心着色
- **智能划分**: 适合需要避免贪心退化的场景，计算开销稍大

### Q6: 如何处理超大规模图（> 10000 节点）？

**A**: 使用 `Main_Multilevel_qaoa_large_graph.py`:
```bash
python Main_Multilevel_qaoa_large_graph.py
```

### Q7: 含噪声 QAOA 的噪声概率如何设置？

**A**: 噪声概率通常设置为 0.01~0.1:
```bash
python run_experiments.py --adapt-noise --noise-prob 0.05
```

### Q8: 如何复现实验结果？

**A**: 设置相同的随机种子:
```bash
python run_experiments.py --adapt --seed 10
```

### Q9: 如何准确率计算？

**A**: 准确率 = (总边数 - 冲突数) / 总边数 = 1 - 冲突数 / 总边数

## 依赖项

```
mindspore >= 2.0
mindquantum >= 0.9
networkx >= 2.8
matplotlib >= 3.5
numpy >= 1.21
pandas >= 1.3
metis >= 0.2.0
```

## 参考资料

1. Farhi, E., et al. "A Quantum Approximate Optimization Algorithm" (2014)
2. [MindQuantum 文档](https://www.mindspore.cn/mindquantum)
3. [MindSpore 文档](https://www.mindspore.cn/)
4. Zhou, L., et al. "Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices" (2020)

## 与天衍平台版本的区别

| 特性 | standard_and_adapt_QAOA | adapt_QAOA_Tianyan |
|------|------------------------|-------------------|
| 框架 | MindQuantum (MindSpore) | cqlib |
| 执行方式 | 本地仿真 | 远程真机/仿真 |
| 配置 | 无需密钥 | 需要 LOGIN_KEY |
| 适用场景 | 算法研究、实验 | 实际量子计算 |
| 量子比特 | 不限（受限于硬件） | 受限于机时包 |
| 数据加载 | graph_loader.py | graph_loader.py（兼容） |

## 扩展开发

### 添加新的 QAOA 变体

1. 创建新的模块文件（如 `multilevel_custom_QAOA_k_coloring.py`）
2. 实现 `solve_k_coloring_custom()` 函数
3. 实现 `sequential_process_subgraphs_custom()` 函数
4. 实现 `iterative_optimization_custom()` 函数
5. 在主程序中添加相应的 `main_custom()` 函数

**模板**:
```python
# multilevel_custom_QAOA_k_coloring.py
def solve_k_coloring_custom(graph, k, p=1, num_steps=1000, ...):
    """
    自定义 QAOA 求解 k 着色
    Returns:
        tuple: (best_k, conv_param, best_coloring, conflict_history, best_params)
    """
    # 实现自定义 QAOA 算法
    pass

def sequential_process_subgraphs_custom(subgraphs, mappings, algo_params, ...):
    """顺序处理子图着色"""
    pass

def iterative_optimization_custom(graph, global_coloring, subgraphs, mappings, ...):
    """迭代优化全局着色"""
    pass
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目仅供学术研究使用。
