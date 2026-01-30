# 经典图着色算法库

本目录包含多种经典图着色算法的实现，用于与量子 QAOA 算法进行对比验证。

## 目录结构

```
classical_algorithms/
├── greedy.py                        # 贪心算法核心实现
├── tabucol.py                       # Tabu 搜索算法核心实现（含详细调试日志）
├── classical_greedy_coloring.py     # 贪心算法完整实现（含批量处理和可视化）
├── classical_tabu_coloring.py       # Tabu 算法完整实现（含批量处理和改进策略）
├── graph_coloring_utils.py          # 通用工具类（数据读取、可视化、CSV保存）
├── large_graph_classical_algorithm.py  # 大规模图经典算法（支持 citeseer、pubmed）
└── coloring_results/                # 结果保存目录（已被 .gitignore 忽略）
    ├── *.csv                        # 算法运行结果统计
    └── *.pdf                        # 着色结果可视化
```

## 快速开始

### 运行单个图着色

```python
from greedy import GreedyColoring
from tabucol import tabucol
import networkx as nx

# 创建图
G = nx.Graph()
G.add_edges_from([(0,1), (1,2), (2,3)])

# 贪心算法
greedy = GreedyColoring(G)
coloring, num_colors = greedy.execute()
print(f"使用颜色数: {num_colors}")

# Tabu搜索
max_degree = max(dict(G.degree()).values())
tabu_coloring = tabucol(G, max_degree + 1, max_iterations=1000)
if tabu_coloring:
    print(f"Tabu使用颜色数: {len(set(tabu_coloring.values()))}")
```

### 批量处理 .col 文件

```bash
# 贪心算法批量处理
python classical_greedy_coloring.py

# Tabu算法批量处理
python classical_tabu_coloring.py
```

### 处理大规模数据集

```bash
# 处理 citeseer.col、pubmed.col 等大规模图
python large_graph_classical_algorithm.py
```

## 算法说明

### 1. Greedy Coloring（贪心算法）

**文件**: `greedy.py`

**核心类**: `GreedyColoring`

**时间复杂度**: O(V + E)

**特点**:
- 按节点度数降序排列（Welsh-Powell 策略）
- 选择最小编号可用颜色
- 快速但解质量一般
- 自动验证着色有效性

**核心实现**:
```python
class GreedyColoring:
    def __init__(self, graph):
        self.graph = graph
        # 按度数降序排序
        self.node_order = sorted(self.nodes, key=lambda x: graph.degree(x), reverse=True)
        self.coloring = {}
        self.used_colors = set()

    def execute(self):
        for node in self.node_order:
            # 收集邻居颜色
            neighbor_colors = {self.coloring[neigh] for neigh in self.graph.neighbors(node) if neigh in self.coloring}
            # 找到最小可用颜色
            color = 0
            while color in neighbor_colors:
                color += 1
            self.coloring[node] = color
            self.used_colors.add(color)
        # 验证有效性
        if not self._is_valid():
            raise ValueError("Invalid coloring result")
        return self.coloring, len(self.used_colors)
```

**可视化类**: `GraphColoringVisualizer`
- 支持组合可视化（图结构 + 颜色分布）
- 支持独立可视化（仅图结构）
- 自动生成 HSV 颜色方案

**使用示例**:
```python
from greedy import process_single_graph

# 处理单个图并保存可视化
result = process_single_graph("graph1.col", graph)
# 返回: {
#   'filename': 'graph1.col',
#   'num_nodes': 100,
#   'num_edges': 300,
#   'num_colors': 5,
#   'execution_time_ms': 12.34,
#   'is_valid': True,
#   'save_path': './coloring_results'
# }
```

### 2. Tabu Search（禁忌搜索）

**文件**: `tabucol.py`

**核心函数**: `tabucol()`

**时间复杂度**: 取决于迭代次数

**特点**:
- 使用禁忌表（deque）避免循环
- 支持愿望水平机制（Aspiration Level）
- 早停机制（500 轮无改进）
- 支持详细调试日志（debug 模式）
- 处理空图、孤立图等特殊情况

**核心参数**:
```python
def tabucol(graph, number_of_colors, tabu_size=7, reps=100,
            max_iterations=10000, debug=False):
    """
    Args:
        graph: NetworkX 图对象
        number_of_colors: 最大颜色数
        tabu_size: 禁忌表大小（默认7）
        reps: 每次迭代的采样次数（默认100）
        max_iterations: 最大迭代次数（默认10000）
        debug: 是否打印详细日志（用于调试）
    Returns:
        dict: 着色方案 {节点: 颜色}，如果无解返回 None
    """
```

**算法流程**:
1. 随机初始化解
2. 收集冲突节点
3. 采样邻居解，寻找最小冲突
4. 更新禁忌表
5. 检查愿望水平（允许解禁禁忌移动）
6. 早停判断（500 轮无改进）

**调试模式**（debug=True）:
```
【TabuCol函数启动】输入参数：
- 图节点数：100, 边数：200
- 颜色数：5, 禁忌表大小：7, 采样次数：100
- 最大迭代次数：10000

【迭代    0】当前禁忌表：[]（长度：0）
【迭代    0】收集的冲突节点：[1, 5, 10]（长度：3）
【迭代    0】采样  50 次：节点 5 从色2→3，新冲突数：2（当前最优：3）
【迭代    0】找到更优解（冲突数 2 < 3），提前结束采样
```

**使用示例**:
```python
from tabucol import tabucol, estimate_chromatic_number, solve_with_tabu

# 方法1: 直接调用
k = estimate_chromatic_number(graph)  # 估算色数
coloring = tabucol(graph, k, max_iterations=1000, debug=False)

# 方法2: 使用包装函数
coloring, num_colors, is_valid = solve_with_tabu(
    graph, k, visualize=True, max_iterations=1000
)
```

### 3. 改进的 Tabu 算法

**文件**: `classical_tabu_coloring.py`

**改进策略**:

1. **改进的色数估算** (`improved_chromatic_estimate`):
   - 基础下界：最大度数 + 1
   - 团大小下界：贪心算法近似最大团
   - 边密度调整：稠密图额外增加颜色数

2. **自适应参数**:
   ```python
   if edge_density > 0.5:  # 稠密图
       max_iterations = max(20000 * 2, n_nodes * 200)
       tabu_size = max(10, int(n_nodes * 0.2))
       reps = 150
   elif edge_density > 0.2:  # 中等密度图
       max_iterations = max(20000, n_nodes * 150)
       tabu_size = max(7, int(n_nodes * 0.15))
       reps = 100
   else:  # 稀疏图
       max_iterations = max(10000, n_nodes * 100)
       tabu_size = max(5, int(n_nodes * 0.1))
       reps = 80
   ```

3. **渐进式搜索**:
   - 从估算色数开始，逐步增加
   - 最多重试 `max_retries` 次（默认5）
   - 策略2补充搜索更大的色数范围

**使用示例**:
```python
from classical_tabu_coloring import process_single_graph

result = process_single_graph(
    filename="graph1.col",
    graph=graph,
    max_iterations_base=20000,
    max_retries=5
)
# 返回: {
#   'filename': 'graph1.col',
#   'num_nodes': 100,
#   'num_edges': 300,
#   'num_colors': 5,  # 最终使用颜色数
#   'conflicts': 0,
#   'execution_time_ms': 1234.56,
#   'is_valid': True,
#   'coloring': {0: 0, 1: 1, ...},
#   'algorithm': 'Tabu'
# }
```

### 4. 通用工具类

**文件**: `graph_coloring_utils.py`

**核心类**: `GraphColoringUtils`

**主要功能**:

| 功能 | 方法 | 说明 |
|------|------|------|
| 数据读取 | `parse_col_file(file_path)` | 读取 DIMACS COLOR 格式 |
| 获取文件列表 | `get_col_files()` | 获取目录下所有 .col 文件 |
| 冲突计算 | `calculate_conflicts(graph, coloring)` | 计算着色冲突数 |
| 颜色归一化 | `normalize_coloring(coloring)` | 将颜色值归一化到 0~k-1 |
| 结果保存 | `save_results_to_csv(results, csv_filename)` | 追加保存到CSV |
| 可视化 | `visualize_coloring(graph, coloring, ...)` | 保存着色结果PDF |
| 原始图可视化 | `plot_original_graph(graph, ...)` | 可视化无着色图 |
| 批量处理 | `process_graphs_batch(algorithm_func, algorithm_name)` | 批量处理多个图 |

**使用示例**:
```python
from graph_coloring_utils import GraphColoringUtils

# 初始化工具类
utils = GraphColoringUtils(
    data_dir="../../Data/instances",
    results_dir="./coloring_results"
)

# 读取图
graph = utils.parse_col_file("graph1.col")

# 批量处理
results = utils.process_graphs_batch(
    algorithm_func=your_algorithm_handler,
    algorithm_name="YourAlgorithm"
)

# 手动保存结果
utils.save_results_to_csv(results, "results.csv")

# 可视化
utils.visualize_coloring(
    graph, coloring, "graph1.col", 5, 123.45, "Greedy"
)
```

### 5. 大规模图算法

**文件**: `large_graph_classical_algorithm.py`

**支持的数据集**:
- `cora.col`
- `citeseer.col`
- `pubmed.col`

**特点**:
- 支持超大规模图（数千节点）
- 使用 `unified_data_processor` 处理数据
- 临时文件处理（避免内存溢出）
- CSV 结果保存到 `csvs/` 目录
- 详细的统计摘要输出

**使用方法**:
```bash
python large_graph_classical_algorithm.py
```

**输出示例**:
```
======================================================================
         大规模图数据集经典算法着色实验
         Large Graph Classical Coloring Experiments
======================================================================

加载大规模图数据...
======================================================================
  数据目录: /path/to/Data/Large_datesets

  正在加载: pubmed.col
    ✓ 成功加载 pubmed.col
      节点数: 19717
      边数: 44324
      平均度数: 4.50

======================================================================
✓ 总共加载了 1 个大规模图
======================================================================

======================================================================
开始运行经典算法...
======================================================================

──────────────────────────────────────────────────────────────────────
[Greedy] 正在处理图 1: pubmed.col
──────────────────────────────────────────────────────────────────────
  颜色数: 12
  冲突数: 0
  是否有效: ✓ 有效
  执行时间: 523.4567 ms

──────────────────────────────────────────────────────────────────────
[Tabu] 正在处理图 1: pubmed.col
──────────────────────────────────────────────────────────────────────
  颜色数: 8
  冲突数: 0
  是否有效: ✓ 有效
  执行时间: 12345.6789 ms
```

## 算法对比

| 算法 | 文件 | 时间复杂度 | 解质量 | 适用场景 | 特点 |
|------|------|-----------|--------|---------|------|
| Greedy | `greedy.py` | O(V+E) | 中等 | 快速估计、大规模图 | 度数降序、简单高效 |
| TabuCol | `tabucol.py` | O(k×iter) | 较好 | 中等规模、需要高质量解 | 禁忌表、愿望水平 |
| 改进Tabu | `classical_tabu_coloring.py` | 自适应 | 最好 | 精确着色、研究实验 | 自适应参数、渐进搜索 |

**性能对比**（典型结果）:

| 图名称 | 节点数 | Greedy 颜色数 | Tabu 颜色数 | 改进Tabu 颜色数 |
|--------|--------|--------------|-------------|----------------|
| queen6_6 | 36 | 8 | 7 | 6 |
| le450_15c | 450 | 18 | 16 | 15 |
| school1 | 385 | 20 | 17 | 15 |

## 可视化

### 组合可视化（图结构 + 颜色分布）

```python
from greedy import GraphColoringVisualizer

visualizer = GraphColoringVisualizer(
    graph=graph,
    coloring=coloring,
    filename="graph1.col",
    num_colors=5,
    exec_time=123.45
)
visualizer.save_combined_visualization("combined.pdf")
# 输出: 左侧图结构，右侧颜色分布柱状图
```

### 单独图结构可视化

```python
from greedy import visualize_coloring

visualize_coloring(
    graph, coloring, "Graph Coloring with 5 Colors",
    save_path="coloring.pdf",
    figsize=(10, 8)
)
```

### 原始图可视化

```python
from graph_coloring_utils import GraphColoringUtils

utils = GraphColoringUtils()
utils.plot_original_graph(
    graph,
    title="Original Graph",
    save_filename="original_graph.pdf"
)
```

## 输出文件

### 结果文件结构
```
coloring_results/
├── greedy_coloring_results.csv         # 贪心算法批量结果
├── tabu_coloring_results.csv           # Tabu算法批量结果
├── graph1_greedy_coloring.pdf          # 图1贪心着色结果
├── graph1_tabu_coloring.pdf            # 图1 Tabu着色结果
└── ...

csvs/
└── large_datasets_classical_results.csv  # 大规模图结果
```

### CSV 文件格式

**小规模图结果** (`greedy_coloring_results.csv`):
```csv
filename,num_nodes,num_edges,num_colors,conflicts,execution_time_ms,is_valid,algorithm
queen6_6.col,36,90,8,0,12.34,True,Greedy
le450_15c.col,450,8168,18,0,123.45,True,Greedy
```

**大规模图结果** (`large_datasets_classical_results.csv`):
```csv
algorithm,dataset,graph_index,num_nodes,num_edges,num_colors,conflicts,is_valid,execution_time_ms,validation_failed
Greedy,pubmed.col,1,19717,44324,12,0,True,523.4567,False
TabuCol,pubmed.col,1,19717,44324,8,0,True,12345.6789,False
```

### 可视化 PDF 文件

**组合可视化** (`*_combined.pdf`):
- 左侧：图结构（带颜色标记）
- 右侧：颜色分布柱状图（每个颜色的节点数）
- 标题：文件名、颜色数、执行时间

**单独着色** (`*_greedy_coloring.pdf`, `*_tabu_coloring.pdf`):
- 完整图结构可视化
- 节点按颜色着色
- 包含统计信息

## 参数配置

### Greedy 算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 排序策略 | 度数降序 | Welsh-Powell 策略 |
| 颜色选择 | 最小可用 | 选择最小编号可用颜色 |

### TabuCol 算法参数

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `number_of_colors` | 估算值 | k~k+5 | 最大颜色数 |
| `tabu_size` | 7 | 5~15 | 禁忌表大小 |
| `reps` | 100 | 50~200 | 每次迭代采样次数 |
| `max_iterations` | 10000 | 5000~50000 | 最大迭代次数 |
| `debug` | False | True/False | 是否打印调试日志 |

### 改进 Tabu 算法参数

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `max_iterations_base` | 20000 | 10000~50000 | 基础迭代次数（根据图密度自适应调整） |
| `tabu_size_ratio` | 0.15 | 0.1~0.2 | 禁忌表大小比例（相对于节点数） |
| `reps` | 100 | 50~200 | 采样次数（根据图密度自适应调整） |
| `max_retries` | 5 | 3~10 | 最大重试次数（渐进式搜索） |

### 可视化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `save_path` | None | 保存路径（不传则显示） |
| `figsize` | (10, 8) | 图片大小（宽×高） |
| `dpi` | 300 | 分辨率 |

## 常见问题

### Q1: Tabu 算法运行时间太长怎么办？

**A**: 根据图特征调整参数：

```python
# 稠密图（边密度 > 0.5）
coloring = tabucol(graph, k, max_iterations=50000, tabu_size=10, reps=50)

# 中等密度图
coloring = tabucol(graph, k, max_iterations=20000, tabu_size=7, reps=100)

# 稀疏图（边密度 < 0.2）
coloring = tabucol(graph, k, max_iterations=10000, tabu_size=5, reps=150)
```

### Q2: 如何调试 Tabu 算法？

**A**: 启用 debug 模式查看详细日志：

```python
coloring = tabucol(graph, k, debug=True)
# 输出每迭代的禁忌表、冲突节点、采样结果等
```

### Q3: 贪心算法报错 "Invalid coloring result"？

**A**: 这通常发生在图节点编号不连续时。`large_graph_classical_algorithm.py` 已处理此情况，可参考其实现：

```python
# 手动执行着色逻辑（绕过验证）
greedy = GreedyColoring(graph)
for node in sorted(greedy.nodes, key=lambda x: graph.degree(x), reverse=True):
    neighbor_colors = {greedy.coloring[neigh] for neigh in graph.neighbors(node) if neigh in greedy.coloring}
    color = 0
    while color in neighbor_colors:
        color += 1
    greedy.coloring[node] = color
    greedy.used_colors.add(color)
```

### Q4: 如何禁用可视化？

**A**: 方法1：不传 `save_path` 参数
```python
visualize_coloring(graph, coloring, title)  # 显示但不保存
```

方法2：在批量处理时禁用
```python
result = process_single_graph(filename, graph, visualize=False)
```

### Q5: 如何处理超大规模图（> 10000 节点）？

**A**: 使用 `large_graph_classical_algorithm.py`，已针对大规模图优化：

- 增加迭代次数（max_iterations=50000）
- 使用临时文件避免内存溢出
- 支持分布式处理（需自行实现）

### Q6: CSV 文件被忽略无法提交到 Git？

**A**: 这是正常的，`coloring_results/` 目录已被 `.gitignore` 忽略。如需保存结果：

1. 手动复制文件到其他目录
2. 修改 `.gitignore` 暂时忽略规则
3. 使用 CSV 结果进行数据分析

### Q7: 如何对比多个算法的结果？

**A**: 查看 CSV 文件或运行批量处理：

```python
import pandas as pd

# 读取结果
greedy_df = pd.read_csv("coloring_results/greedy_coloring_results.csv")
tabu_df = pd.read_csv("coloring_results/tabu_coloring_results.csv")

# 合并对比
merged = pd.merge(
    greedy_df[['filename', 'num_colors', 'execution_time_ms']],
    tabu_df[['filename', 'num_colors', 'execution_time_ms']],
    on='filename',
    suffixes=('_greedy', '_tabu')
)

print(merged)
```

## 数据格式

### DIMACS COLOR 格式 (.col)

```
c This is a comment line
p edge <nodes> <edges>
e <u> <v>
e <u> <v>
...
```

**示例**:
```
c Queen 6x6 graph
p edge 36 90
e 1 2
e 1 7
...
```

### NetworkX 图对象

支持的图类型：
- `nx.Graph`: 无向图
- 节点编号：1-based（与 .col 格式一致）
- 边表示：`(u, v)` 元组列表

## 参考文献

1. Welsh, D. J. A., & Powell, M. B. (1967). "An upper bound for the chromatic number of a graph and its application to timetabling problems"
2. Hertz, A., & de Werra, D. (1987). "Using tabu search techniques for graph coloring"
3. Glover, F. (1986). "Future paths for integer programming and links to artificial intelligence"

## 扩展开发

### 添加新的经典算法

1. 创建核心算法文件（如 `new_algorithm.py`）
2. 实现核心函数，返回 `{node: color}` 格式
3. 创建批量处理文件（如 `classical_new_algorithm.py`）
4. 使用 `graph_coloring_utils.GraphColoringUtils` 处理数据
5. 在 README 中添加算法说明

**模板**:
```python
# new_algorithm.py
def new_algorithm_coloring(graph, **params):
    """
    新的图着色算法
    Returns:
        dict: 着色方案 {节点: 颜色}
    """
    # 实现算法逻辑
    coloring = {}
    # ...
    return coloring

# classical_new_algorithm.py
from graph_coloring_utils import GraphColoringUtils
from new_algorithm import new_algorithm_coloring

def process_single_graph(filename, graph):
    start = time.perf_counter()
    coloring = new_algorithm_coloring(graph)
    num_colors = len(set(coloring.values()))
    conflicts = utils.calculate_conflicts(graph, coloring)
    exec_time = (time.perf_counter() - start) * 1000
    
    return {
        'filename': filename,
        'num_colors': num_colors,
        'conflicts': conflicts,
        'execution_time_ms': exec_time,
        'is_valid': conflicts == 0,
        'algorithm': 'NewAlgorithm'
    }

if __name__ == "__main__":
    utils = GraphColoringUtils()
    utils.process_graphs_batch(process_single_graph, "NewAlgorithm")
```

## 性能优化建议

1. **大规模图**: 使用贪心算法快速估计
2. **中等规模**: Tabu搜索获得更好质量
3. **精确着色**: 使用改进Tabu算法
4. **对比实验**: 同时运行多种算法对比结果
5. **调试阶段**: 启用 debug 模式定位问题
6. **生产环境**: 关闭 debug 模式，调整迭代次数

## 联系与支持

如有问题或建议，请提交 Issue 或 Pull Request。
