# HQAOA - 分层量子近似优化算法图着色

本项目实现了基于多层次 QAOA (Hierarchical QAOA) 的图着色算法，适配天衍量子计算平台和 MindSpore 量子框架，并提供经典算法对比。

## 项目结构

```
HQAOA_coloring/
├── HAdaQAOA/
│   └── HadaQAOA/
│       ├── adapt_QAOA_Tianyan/          # 天衍平台 QAOA 实现
│       │   └── QAOA/
│       │       ├── config_template.py    # 配置模板
│       │       ├── Main_*.py            # 主程序（多种版本）
│       │       ├── multilevel_*.py      # 核心算法实现
│       │       └── train_params_*.py    # 参数训练模块
│       ├── classical_algorithms/         # 经典算法对比
│       │   ├── greedy.py               # 贪心算法
│       │   ├── tabucol.py              # Tabu 搜索
│       │   ├── graph_coloring_utils.py # 工具函数
│       │   └── large_graph_classical_algorithm.py
│       ├── standard_and_adapt_QAOA/     # 标准与自适应 QAOA
│       │   ├── multilevel_common.py    # 公共函数库
│       │   ├── multilevel_adapt_QAOA_k_coloring.py
│       │   ├── multilevel_standard_QAOA_k_coloring.py
│       │   ├── multilevel_adapt_noise_QAOA_k_coloring.py
│       │   ├── Main_*.py               # 主程序
│       │   └── run_experiments.py      # 实验运行脚本
│       └── Data/                        # 数据目录（不上传）
├── .gitignore                         # Git 忽略规则
└── README.md                          # 本文件
```

## 功能特性

### 1. 量子算法

#### MindSpore 版本 (standard_and_adapt_QAOA)

- **多层次 QAOA (HQAOA)**: 通过图划分处理大规模图
- **标准 QAOA**: 使用固定的 X 门作为混合算子
- **自适应 QAOA**: 梯度选择最优混合算子
- **含噪声 QAOA**: 模拟退极化噪声，研究噪声影响

#### 天衍平台版本 (adapt_QAOA_Tianyan)

- **多层次 QAOA**: 适配天衍量子计算平台
- **真机执行**: 支持实际量子计算机运行
- **参数训练**: 线上参数优化
- **智能划分**: 根据量子比特约束自动调整子图大小
- **稀疏图优化**: DSatur 自动寻优贪心算法

> **天衍平台**: [中电信量子计算平台](https://qc.zdxlz.com/home?lang=zh)
> **MindSpore**: [华为昇思量子框架](https://www.mindspore.cn/)

### 2. 经典算法

- **贪心算法**: 快速估计，度数降序策略
- **Tabu 搜索**: 高质量解，禁忌表优化
- **Welch-Powell**: 简单高效

### 3. 算法对比

- 量子 vs 经典算法性能对比
- 不同规模图的适用性分析
- 可视化结果展示

## 快速开始

### 选择版本

- **MindSpore 版本**: 适合本地仿真、算法研究
- **天衍平台版本**: 适合实际量子计算、真机执行

### 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd HQAOA_coloring

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置天衍平台（仅天衍版本需要）

```bash
cd HAdaQAOA/HadaQAOA/adapt_QAOA_Tianyan/QAOA
cp config_template.py config.py

# 编辑 config.py，填入登录密钥
```

**注意**: `config.py` 文件已被 `.gitignore` 忽略，不会上传到 GitHub。

### 3. 运行示例

#### MindSpore 版本

```bash
cd ../../standard_and_adapt_QAOA

# 运行自适应 QAOA
python Main_Multilevel_qaoa.py

# 运行标准 QAOA
python Main_Multilevel_qaoa.py

# 运行含噪声 QAOA
python Main_Multilevel_qaoa.py

# 使用实验脚本
python run_experiments.py --adapt --standard
```

#### 天衍平台版本

```bash
cd ../adapt_QAOA_Tianyan/QAOA

# 朴素划分版本
python Main_Multilevel_qaoa_tianyan.py --format-type col

# 智能划分版本
python Main_Multilevel_qaoa_tianyan_smart.py --format-type col

# 大规模图版本
python Main_Multilevel_qaoa_tianyan_large.py --max-qubits 200
```

#### 经典算法

```bash
cd ../../classical_algorithms

# 运行大规模图经典算法对比
python large_graph_classical_algorithm.py

# 贪心算法批量处理
python classical_greedy_coloring.py

# Tabu 算法批量处理
python classical_tabu_coloring.py
```

## 文档

- [MindSpore 版本详细说明](HAdaQAOA/HadaQAOA/standard_and_adapt_QAOA/README.md)
- [天衍平台版本详细说明](HAdaQAOA/HadaQAOA/adapt_QAOA_Tianyan/QAOA/README.md)
- [经典算法详细说明](HAdaQAOA/HadaQAOA/classical_algorithms/README.md)
- [安全配置说明](GITHUB_SECURITY_NOTES.md)

## 算法说明

### 多层次 QAOA

1. **图划分**: 将大图划分为多个子图
   - 朴素划分: `divide_graph(num_subgraphs, max_nodes)`
   - 智能划分: `smart_divide_graph_with_qubit_constraint(max_qubits)`

2. **子图着色**: 使用 QAOA 对每个子图进行着色
   - 孤立节点: 批量处理，所有节点同色
   - 环图: 专用算法（偶环 2 色，奇环 3 色）
   - 完全图: 贪心着色
   - 普通图: QAOA + 贪心混合策略

3. **合并优化**: 合并子图着色结果并优化边界节点
   - 统一子图颜色编号
   - 处理边界节点的颜色冲突
   - 迭代改进全局着色

### 经典算法

详见 [classical_algorithms/README.md](HAdaQAOA/HadaQAOA/classical_algorithms/README.md)

## 数据集

### 支持的数据格式

- `.col` - DIMACS COLOR 格式
- `.pkl` - Pickle 序列化格式（NetworkX Graph）

### 示例数据集

| 数据集 | 节点数 | 边数 | 边密度 | 适用版本 |
|--------|--------|------|--------|----------|
| queen6_6 | 36 | 90 | 0.148 | 所有版本 |
| le450_15c | 450 | 8168 | 0.081 | MindSpore、天衍 |
| cora.col | 2708 | 5429 | 0.00148 | 天衍大规模 |
| citeseer.col | 3312 | 4660 | 0.00085 | 天衍大规模 |
| pubmed.col | 19717 | 44338 | 0.00023 | 天衍大规模 |

**注意**: 数据文件需要放置在相应的 `Data/` 目录下。

**数据集信息**: 
- 小规模图数据集可上传到 GitHub
- 大规模图数据集（> 100MB）建议使用独立存储

## 输出说明

### MindSpore 版本

- **日志文件**: `logs/*.log`
- **结果文件**: `csvs/*.csv`
- **可视化**: `graph_visualizations/`, `subgraph_visualizations/`, `experiment_visualizations/`

### 天衍平台版本

- **日志文件**: `logs/*.log`
- **结果文件**: `csvs/*.csv`
- **可视化**: `large_graph_visualizations/`, `large_subgraph_visualizations/`

### 经典算法

- **结果文件**: `coloring_results/*.csv`, `csvs/*.csv`
- **可视化**: `coloring_results/*.pdf`

**注意**: 所有输出文件和可视化目录已被 `.gitignore` 忽略。

## 命令行参数

### 天衍平台版本

```bash
--graph-dir <path>      # 图数据目录
--format-type <type>    # 数据格式 (auto/col/pkl)
--seed <int>           # 随机种子 (默认: 10)
--train-params         # 是否训练参数
--train-max-iter <int> # 最大迭代次数 (默认: 50)
--train-lr <float>     # 学习率 (默认: 0.05)
--max-qubits <int>     # 量子比特数限制 (默认: 36/200)
```

### MindSpore 版本

```bash
--adapt               # 运行自适应 QAOA
--standard            # 运行标准 QAOA
--adapt-noise         # 运行含噪声 QAOA
--noise-prob <float> # 噪声概率 (默认: 0.05)
--seed <int>          # 随机种子 (默认: 10)
--graph-dir <path>    # 图数据目录
--format-type <type>  # 数据格式 (auto/col/pkl)
```

## 依赖项

### MindSpore 版本

```
mindspore >= 2.0
mindquantum >= 0.9
networkx >= 2.8
matplotlib >= 3.5
numpy >= 1.21
pandas >= 1.3
metis >= 0.2.0
```

### 天衍平台版本

```
cqlib >= 0.1.0
networkx >= 2.8
matplotlib >= 3.5
numpy >= 1.21
pandas >= 1.3
metis >= 0.2.0
```

### 经典算法

```
networkx >= 2.8
matplotlib >= 3.5
numpy >= 1.21
pandas >= 1.3
```

## 性能对比

### 算法对比

| 算法 | 混合算子 | 参数数量 | 解质量 | 计算开销 | 适用场景 |
|------|----------|----------|--------|----------|----------|
| 标准 QAOA | 固定 X 门 | 少 | 中等 | 低 | 基准测试、快速实验 |
| 自适应 QAOA | 梯度选择 | 多 | 好 | 高 | 需要高质量解 |
| 含噪声 QAOA | 梯度选择 | 多 | 中等 | 高 | 真实设备模拟 |
| 贪心算法 | N/A | N/A | 中等 | 很低 | 快速估计、大规模图 |
| Tabu 搜索 | N/A | N/A | 较好 | 中等 | 中等规模、高质量解 |

### 典型结果对比

| 图名称 | 节点数 | Greedy | Tabu | 标准 QAOA | 自适应 QAOA |
|--------|--------|--------|------|-----------|-------------|
| queen6_6 | 36 | 8 | 7 | 7 | 6 |
| le450_15c | 450 | 18 | 16 | 16 | 15 |
| school1 | 385 | 20 | 17 | 17 | 15 |

## 安全说明

**重要**: 本项目使用天衍平台需要配置登录密钥，相关配置文件已被 `.gitignore` 忽略。

### 上传前检查

1. ✅ 创建 `config.py` 文件（从 `config_template.py` 复制）
2. ✅ 在 `config.py` 中填入实际的登录密钥
3. ✅ 确认 `config.py` 已被 `.gitignore` 忽略

详细安全配置请参考 [GITHUB_SECURITY_NOTES.md](GITHUB_SECURITY_NOTES.md)。

## 常见问题

### Q: 如何选择 MindSpore 版本还是天衍平台版本？

A: 
- **MindSpore 版本**: 适合本地仿真、算法研究，无需配置密钥
- **天衍平台版本**: 适合实际量子计算、真机执行，需要配置登录密钥

### Q: 如何处理超大规模图（> 10000 节点）？

A: 使用天衍平台的大规模图版本：
```bash
python Main_Multilevel_qaoa_tianyan_large.py --max-qubits 200
```

### Q: 如何禁用可视化？

A: 修改代码中的 `VISUALIZATION_CONFIG` 或注释掉可视化相关函数调用。

### Q: 运行时间过长怎么办？

A: 减少以下参数：
- 减小 `max_k` (最大颜色数)
- 减小 `p` (QAOA 层数)
- 减小 `num_steps` (最大训练步数)
- 减小 `max_iter` (迭代优化次数)

## 参考文献

本项目算法基于以下论文：

1. Hierarchical Quantum Approximate Optimization Algorithm for Graph Coloring
   - DOI: [10.48550/arXiv.2504.21335](https://doi.org/10.48550/arXiv.2504.21335)

2. Enhanced Adaptive QAOA for Large-Scale Graph Problems
   - DOI: [10.1103/PhysRevA.112.052435](https://doi.org/10.1103/l5w7-x27x)

3. Farhi, E., et al. (2014). "A Quantum Approximate Optimization Algorithm"
4. Zhou, L., et al. (2020). "Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices"
5. Hertz, A., & de Werra, D. (1987). "Using tabu search techniques for graph coloring"
6. Welsh, D. J. A., & Powell, M. B. (1967). "An upper bound for the chromatic number of a graph and its application to timetabling problems"

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目仅供学术研究使用。
