"""
共享模块：多层次QAOA图着色通用函数库

本模块提供多层次QAOA图着色算法的共享函数，支持三种算法变体：
- 标准QAOA (Standard QAOA): 固定混合算子
- 自适应QAOA (Adaptive QAOA): 梯度选择混合算子
- 含噪声自适应QAOA (Adaptive QAOA with Noise): 退极化噪声模拟

函数分类：
1. 配置与初始化：路径、日志、编码转换
2. 图处理工具：子图划分、图类型判断、图签名
3. QAOA核心组件：哈密顿量构建、线路生成、混合器池
4. 着色辅助函数：冲突计算、贪心着色、环图着色
5. 可视化工具：原图、子图、着色结果展示
6. 异常处理：统一异常处理函数
"""

# ==============================================================================
# 0. 导入模块与配置
# ==============================================================================

import copy
import time
import traceback
import json
import os
import hashlib
import metis
import re
import logging
import csv
from math import log2, ceil
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import mindspore
from mindspore import nn, Tensor
from mindquantum import Circuit, ParameterResolver, MQAnsatzOnlyLayer, H, UN,  GlobalPhase, commutator
from mindquantum.simulator import Simulator
from mindquantum.core.operators import TimeEvolution, Hamiltonian, QubitOperator
from mindquantum.core.gates import DepolarizingChannel, Rzz, Rxx, Ryy, Ryz, Rxy, Rxz,RX,RY,Measure
import mindspore as ms
import numpy as np
import networkx as nx
import matplotlib
# 使用非交互式后端，图片显示后不阻塞程序继续执行
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# 统一日志目录路径：使用绝对路径指向HadaQAOA/logs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
mindspore.set_device('CPU')

# ---------- 补丁：numpy → python 原生类型 ----------
# 让 json 遇到 numpy 标量/数组时自动转原生类型
json._default = json.JSONEncoder().default
json.JSONEncoder.default = lambda self, obj: (
    obj.item() if isinstance(obj, (np.integer, np.floating)) else
    obj.tolist() if isinstance(obj, np.ndarray) else
    json._default(self, obj)
)


# ==============================================================================
# 1. 配置与初始化函数
# ==============================================================================

def setup_logger(dataset, graph_id):
    """
    初始化日志系统，为每个数据集和图创建独立的日志文件
    
    参数:
        dataset: 数据集名称
        graph_id: 图的ID索引
    
    返回:
        logging.Logger: 配置好的日志记录器
    """
    log_dir = LOGS_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{dataset}_graph_{graph_id}.log")

    # 避免重复添加处理器
    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(f"subgraph_processor_{dataset}_{graph_id}")



# ==============================================================================
# 2. 图处理工具函数
# ==============================================================================

def divide_graph(graph, num_subgraphs, Q=20, max_nodes=10):
    """
    使用 METIS 划分图，并对大子图递归二分直至节点数 ≤ max_nodes
    
    算法流程：
    1. 使用 METIS 初步划分为 num_subgraphs 个子图
    2. 对每个 ≥ max_nodes 节点的子图递归二分
    3. 保证最终每个子图节点数 ≤ max_nodes（默认10）
    
    参数:
        graph: 待划分的图（networkx图对象）
        num_subgraphs: 目标子图数量
        Q: 图划分平衡因子（默认20）
        max_nodes: 子图最大节点数限制（默认10）
    
    返回:
        tuple: (subgraphs, mappings)
            - subgraphs: 子图列表
            - mappings: 原始节点ID到子图内部节点ID的映射列表
    """
    # ---------- 1. 先正常 METIS 划分 ----------
    if num_subgraphs >= len(graph.nodes):
        subgraphs = [graph.subgraph([node]) for node in graph.nodes]
        mappings = [{node: 0} for node in graph.nodes]
        while len(subgraphs) < num_subgraphs:
            subgraphs.append(nx.Graph())
            mappings.append({})
        return subgraphs, mappings

    # 标准化节点ID为0~n-1
    if min(graph.nodes) != 0:
        mapping = {node: i for i, node in enumerate(graph.nodes)}
        graph = nx.relabel_nodes(graph, mapping)
        reverse_mapping = {v: k for k, v in mapping.items()}
    else:
        reverse_mapping = {n: n for n in graph.nodes}

    # METIS划分
    _, part = metis.part_graph(graph, num_subgraphs, objtype="cut", ufactor=200)
    subgraph_nodes = {i: [] for i in range(num_subgraphs)}
    for node, comm_id in enumerate(part):
        subgraph_nodes[comm_id].append(reverse_mapping[node])

    # ---------- 2. 后处理：把 ≥ max_nodes 节点的分量再二分 ----------
    final_subgraphs = []
    final_mappings = []

    def split_until_small(g):
        """递归二分直到节点数 ≤ max_nodes"""
        if g.number_of_nodes() <= max_nodes:
            final_subgraphs.append(g)
            final_mappings.append({old: new for new, old in enumerate(g.nodes)})
            return
        # 否则继续 METIS 二分
        if g.number_of_nodes() >= 2:
            _, part2 = metis.part_graph(g, 2, objtype="cut", ufactor=200)
            nodes0 = [n for n, p in zip(g.nodes, part2) if p == 0]
            nodes1 = [n for n, p in zip(g.nodes, part2) if p == 1]
            split_until_small(g.subgraph(nodes0).copy())
            split_until_small(g.subgraph(nodes1).copy())
        else:
            # 单节点无法二分，直接收下
            final_subgraphs.append(g)
            final_mappings.append({old: new for new, old in enumerate(g.nodes)})

    for comm_id in range(num_subgraphs):
        nodes = subgraph_nodes[comm_id]
        if len(nodes) <= max_nodes:
            sub = graph.subgraph(nodes)
            final_subgraphs.append(sub)
            final_mappings.append({old: new for new, old in enumerate(sub.nodes)})
        else:
            sub = graph.subgraph(nodes)
            split_until_small(sub)

    # ---------- 3. 补齐空社区（保持接口兼容） ----------
    while len(final_subgraphs) < num_subgraphs:
        final_subgraphs.append(nx.Graph())
        final_mappings.append({})

    return final_subgraphs, final_mappings


def is_complete_graph(graph):
    """
    判断图是否为完全图（任意两节点之间都有边）

    参数:
        graph: networkx图对象

    返回:
        bool: 是否为完全图
    """
    n = graph.number_of_nodes()
    if n <= 1:
        return True
    expected_edges = n * (n - 1) // 2
    return graph.number_of_edges() == expected_edges


def smart_divide_graph_with_qubit_constraint(
    graph,
    max_qubits=20,
    max_k_per_subgraph=5,
    Q=20
):
    """
    智能子图划分: 根据量子比特约束自动调整子图大小

    策略:
    1. 根据max_k计算每节点需要的比特数: bits_per_node = ceil(log2(max_k))
    2. 计算最大允许节点数: max_nodes = max_qubits // bits_per_node
    3. 初始子图数: min(num_subgraphs, ceil(total_nodes / max_nodes))
    4. 递归二分超限子图

    参数:
        graph: 待划分的图
        max_qubits: 最大量子比特数 (默认20)
        max_k_per_subgraph: 每个子图最大颜色数 (默认5)
        Q: METIS平衡因子 (默认20)

    返回:
        tuple: (subgraphs, mappings, info)
            - subgraphs: 子图列表
            - mappings: 节点映射列表
            - info: 划分信息字典
    """
    info = {
        'original_nodes': graph.number_of_nodes(),
        'original_edges': graph.number_of_edges(),
        'max_qubits': max_qubits,
        'max_k_per_subgraph': max_k_per_subgraph,
        'subgraph_stats': []
    }

    if graph.number_of_nodes() == 0:
        return [], [], info

    # 计算每节点比特数和最大节点数
    bits_per_node = math.ceil(math.log2(max_k_per_subgraph))
    max_nodes_per_subgraph = max_qubits // bits_per_node

    # 计算需要的初始子图数
    min_subgraphs_needed = math.ceil(graph.number_of_nodes() / max_nodes_per_subgraph)

    # 标准化节点ID
    if min(graph.nodes) != 0:
        mapping = {node: i for i, node in enumerate(graph.nodes)}
        graph_normalized = nx.relabel_nodes(graph, mapping)
        reverse_mapping = {v: k for k, v in mapping.items()}
    else:
        graph_normalized = graph
        reverse_mapping = {n: n for n in graph.nodes}

    # METIS初始划分
    try:
        _, part = metis.part_graph(
            graph_normalized,
            min_subgraphs_needed,
            objtype="cut",
            ufactor=Q * 10
        )
    except Exception as e:
        logging.warning(f"METIS失败，使用简单划分: {e}")
        part = [i % min_subgraphs_needed for i in range(len(graph.nodes))]

    subgraph_nodes = {i: [] for i in range(min_subgraphs_needed)}
    for node, comm_id in enumerate(part):
        subgraph_nodes[comm_id].append(reverse_mapping[node])

    # 递归细化子图
    final_subgraphs = []
    final_mappings = []

    def refine_subgraph(g, mapping, depth=0):
        """递归细化子图直到满足量子比特约束"""
        n = g.number_of_nodes()
        required_qubits = n * bits_per_node

        if n <= max_nodes_per_subgraph or n <= 3 or depth >= 10:
            final_subgraphs.append(g)
            final_mappings.append(mapping)

            info['subgraph_stats'].append({
                'nodes': n,
                'edges': g.number_of_edges(),
                'qubits_required': required_qubits,
                'bits_per_node': bits_per_node,
                'feasible': required_qubits <= max_qubits
            })
            return

        # 需要二分
        logging.info(f"子图{n}节点需要{required_qubits}比特 > {max_qubits}，进行二分(深度{depth})")

        try:
            _, part2 = metis.part_graph(g, 2, objtype="cut", ufactor=Q * 10)
        except Exception:
            # METIS失败，简单二分
            nodes_list = list(g.nodes)
            mid = len(nodes_list) // 2
            part2 = [0] * mid + [1] * (len(nodes_list) - mid)

        nodes0 = [n for n, p in zip(g.nodes, part2) if p == 0]
        nodes1 = [n for n, p in zip(g.nodes, part2) if p == 1]

        refine_subgraph(g.subgraph(nodes0).copy(),
                      {old: new for new, old in enumerate(nodes0)},
                      depth + 1)
        refine_subgraph(g.subgraph(nodes1).copy(),
                      {old: new for new, old in enumerate(nodes1)},
                      depth + 1)

    # 细化所有初始子图
    for comm_id in range(min_subgraphs_needed):
        nodes = subgraph_nodes[comm_id]
        if len(nodes) == 0:
            continue
        sub = graph.subgraph(nodes)
        refine_subgraph(sub, {old: new for new, old in enumerate(sub.nodes)})

    info['num_subgraphs'] = len(final_subgraphs)
    info['max_nodes_per_subgraph'] = max_nodes_per_subgraph
    info['bits_per_node'] = bits_per_node

    feasible_count = sum(1 for s in info['subgraph_stats'] if s['feasible'])
    avg_qubits = sum(s['qubits_required'] for s in info['subgraph_stats']) / len(info['subgraph_stats']) if info['subgraph_stats'] else 0

    logging.info(f"[智能划分] 完成: {graph.number_of_nodes()}节点 → {len(final_subgraphs)}个子图")
    logging.info(f"[智能划分] 可行子图: {feasible_count}/{len(final_subgraphs)}")
    logging.info(f"[智能划分] 平均比特数: {avg_qubits:.1f}")

    return final_subgraphs, final_mappings, info


def is_odd_cycle(graph):
    """
    判断图是否为奇环（奇数个节点的环）
    
    参数:
        graph: networkx图对象
    
    返回:
        bool: 是否为奇环
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    if n != m:
        return False
    if any(graph.degree(node) != 2 for node in graph.nodes):
        return False
    return n % 2 == 1


def is_cycle_graph(graph):
    """
    判断图是否为环图（所有节点度数均为2的连通图）
    
    参数:
        graph: networkx图对象
    
    返回:
        bool: 是否为环图
    """
    n = graph.number_of_nodes()
    if n < 3:
        return False
    if graph.number_of_edges() != n:
        return False
    if not all(graph.degree(node) == 2 for node in graph.nodes):
        return False
    if not nx.is_connected(graph):
        return False
    return True


def get_graph_signature(graph):
    """
    生成图的唯一签名，用于检测同构子图
    
    基于图的结构特征生成MD5哈希签名：
    - 节点数、边数
    - 度序列
    - 规范化的边列表
    
    参数:
        graph: networkx图对象
    
    返回:
        str: 图的MD5哈希签名
    """
    # 规范化节点标签（按度排序后重新编号）
    nodes = sorted(graph.nodes, key=lambda x: (graph.degree(x), x))
    node_mapping = {old: i for i, old in enumerate(nodes)}
    normalized_graph = nx.relabel_nodes(graph, node_mapping)

    # 提取图结构特征
    nodes_sorted = sorted(normalized_graph.nodes)
    edges_sorted = tuple(sorted((u, v) for u, v in normalized_graph.edges))
    degree_sequence = tuple(sorted(normalized_graph.degree(n) for n in nodes_sorted))

    # 生成哈希签名
    signature_data = {
        'num_nodes': normalized_graph.number_of_nodes(),
        'num_edges': normalized_graph.number_of_edges(),
        'degree_sequence': degree_sequence,
        'edges': edges_sorted
    }

    signature_str = json.dumps(signature_data, sort_keys=True).encode()
    return hashlib.md5(signature_str).hexdigest()


# ==============================================================================
# 3. QAOA核心组件函数
# ==============================================================================

# ---------- 3.1 哈密顿量构建函数 ----------

def build_hamiltonian(graph, k, vertex_colors=None, nodes_to_recolor=None, edge_weight= 1000):
    """
    QAOA哈密顿量构建函数

    哈密顿量公式：
    H_edge^(u,v) = sum_{i=0}^{m-1} (I + Z_{u,i} Z_{v,i}) / 2

    期望值范围：
    - 最优状态（颜色不同，Z_u Z_v = -1）：H = 0（能量最低）
    - 最差状态（颜色相同，Z_u Z_v = +1）：H = edge_weight（能量最高）

    注意：MindQuantum 的 get_expectation_with_grad 返回负期望值，
    训练时需要取绝对值：train_value = abs(raw_train_value)

    参数:
        graph: 待着色的图
        k: 颜色数量
        vertex_colors: 字典{节点:目标颜色}，指定节点的目标颜色（保留兼容性）
        nodes_to_recolor: 需要重新着色的节点列表
        edge_weight: 边约束权重

    返回:
        Hamiltonian: QAOA哈密顿量对象
    """
    # 1. 基础参数计算
    n_qubits = max(1, math.ceil(math.log2(k)))
    if nodes_to_recolor is None:
        nodes_to_recolor = list(graph.nodes)
    nodes_to_recolor = [int(node) for node in nodes_to_recolor if node in graph.nodes]

    if not nodes_to_recolor:
        raise ValueError("nodes_to_recolor中无有效节点")

    ham = QubitOperator()

    # 2. 边约束：相邻节点颜色不同
    # H_edge^(u,v) = sum_{i=0}^{m-1} (I + Z_{u,i} Z_{v,i}) / 2
    for u, v in graph.edges:
        u, v = int(u), int(v)
        if u not in nodes_to_recolor or v not in nodes_to_recolor:
            continue

        for i in range(n_qubits):
            # (I + Z_u Z_v) / 2 = 0.5 * I + 0.5 * Z_u Z_v
            identity_term = 0.5 * QubitOperator()
            zz_term = 0.5 * QubitOperator(f"Z{u * n_qubits + i} Z{v * n_qubits + i}")
            ham += edge_weight * (identity_term + zz_term)

    # 检查哈密顿量是否为空
    if not ham.terms:
        raise ValueError("构建的哈密顿量为空，请检查约束条件和图的设置。")

    return Hamiltonian(ham)


# ---------- 3.2 标准QAOA专用函数 ----------


def qaoa_ansatz_standard(graph, k, p=1, vertex_colors=None, nodes_to_recolor=None, edge_weight= 1000):
    """
    标准QAOA线路生成器
    
    特点：
    - 使用固定的混合算子（X门）
    - 每层的混合算子相同
    
    参数:
        graph: 待着色的图
        k: 颜色数量
        p: QAOA层数（电路深度）
        vertex_colors: 顶点颜色字典
        nodes_to_recolor: 需要重新着色的节点列表
        edge_weight: 边约束权重（默认100）
    
    返回:
        Circuit: QAOA量子线路
    """
    n_qudits = ceil(log2(k))
    num_qubits = len(graph.nodes) * n_qudits
    circuit = Circuit()
    circuit += UN(H, list(range(num_qubits)))
    for layer in range(p):
        ham = build_hamiltonian(graph, k, vertex_colors, nodes_to_recolor, edge_weight)
        gamma = ParameterResolver({f'gamma_{layer}': 1.0})
        circuit += TimeEvolution(ham.hamiltonian, gamma).circuit
        beta = ParameterResolver({f'beta_{layer}': 1.0})
        for u in graph.nodes:
            for i in range(n_qudits):
                circuit += RX(f'beta_{layer}').on(int(u * n_qudits + i))
    return circuit





def adapt_qaoa_ansatz(graph, k, p=1, vertex_colors=None, nodes_to_recolor=None, edge_weight= 1000, learning_rate=0.1, adam_steps=50, verbose=False):
    """
    自适应QAOA线路生成器，通过梯度选择最佳混合算子
    
    自适应策略：
    - 每层通过梯度分析选择最优混合算子
    - 梯度最大的混合算子被选中用于当前层
    
    参数:
        graph: 待着色的图
        k: 颜色数量
        p: 线路层数
        vertex_colors: 顶点颜色字典
        nodes_to_recolor: 需要重新着色的节点列表
        edge_weight: 边约束权重（默认100）
        learning_rate: 学习率（未使用）
        adam_steps: Adam步数（未使用）
        verbose: 是否输出详细信息
    
    返回:
        Circuit: 自适应QAOA量子线路
    """
    # 计算每个节点需要的量子比特数和总量子比特数
    n_qudits = math.ceil(math.log2(k))
    num_qubits = len(graph.nodes) * n_qudits

    # 初始化量子线路（均匀叠加态）
    circuit = Circuit()
    circuit += UN(H, list(range(num_qubits)))
    circuit.barrier()

    # 构建目标哈密顿量Hc
    hc = build_hamiltonian(graph, k, vertex_colors, nodes_to_recolor, edge_weight)
    if verbose:
        print(f"构建哈密顿量完成 - 项数: {len(hc.hamiltonian.terms)}, 总量子比特数: {num_qubits}")

    # 生成混合器池
    mixer_pool = build_mixer_pool(num_qubits)
    if verbose:
        print(f"生成混合器池 - 大小: {len(mixer_pool)}")

    # 自适应迭代构建线路
    theta = {}
    mixers_used = []

    for layer in range(p):
        if verbose:
            print(f"\n===== 第 {layer + 1}/{p} 层 =====")
            print(f"当前参数: {theta}")

        # 复制当前线路并应用已有参数
        current_circ = copy.deepcopy(circuit).apply_value(theta)

        # 添加哈密顿量演化层（临时参数用于梯度计算）
        qaoa_cost(current_circ, hc.hamiltonian, 0.01, n_qudits)

        # 计算所有混合器的梯度
        gradients = []
        if verbose:
            print(f"计算 {len(mixer_pool)} 个混合器的梯度...")

        for i, mixer in enumerate(mixer_pool):
            try:
                if verbose and i % 10 == 0 and i > 0:
                    print(f"已计算 {i}/{len(mixer_pool)} 个混合器")

                grad = derivative(
                    hc.hamiltonian,
                    num_qubits,
                    mixer,
                    current_circ,
                    n_qudits
                )
                gradients.append(grad)
            except Exception as e:
                if verbose:
                    print(f"计算混合器 {mixer} 梯度失败: {e}")
                gradients.append(0.0)

        # 选择梯度最大的混合器
        best_idx = np.argmax(gradients)
        best_mixer = mixer_pool[best_idx]
        best_grad = gradients[best_idx]
        mixers_used.append(best_mixer)

        if verbose:
            print(f"选择最佳混合器: {best_mixer} (梯度值: {best_grad:.6f})")

        # 添加新的演化层和混合层
        gamma_param = f"gamma_{layer}"
        beta_param = f"beta_{layer}"

        qaoa_cost(circuit, hc.hamiltonian, gamma_param, n_qudits)
        qaoa_mixer(circuit, best_mixer, beta_param, num_qubits)

    if verbose:
        print("\n===== 自适应QAOA线路构建完成 =====")
        print(f"总线路门数量: {len(circuit)}")
        print(f"使用的混合器序列: {mixers_used}")

    return circuit


# ---------- 3.3 混合器池函数 ----------

def mixer_pool_single(qubits):
    """
    生成单比特和全比特的混合器操作池
    
    包含：
    - 单比特 X 操作：X0, X1, ..., X{n-1}
    - 单比特 Y 操作：Y0, Y1, ..., Y{n-1}
    - 全比特 X 操作：XXX...（n个X）
    - 全比特 Y 操作：YYY...（n个Y）
    
    参数:
        qubits: 总量子比特数
    
    返回:
        list: 混合器操作字符串列表
    """
    pool = []

    # 单比特 X 操作
    single_X = [f'X{i}' for i in range(qubits)]
    pool.extend(single_X)

    # 全比特 X 操作
    all_X = 'X' * qubits
    pool.append(all_X)

    # 单比特 Y 操作
    single_Y = [f'Y{i}' for i in range(qubits)]
    pool.extend(single_Y)

    # 全比特 Y 操作
    all_Y = 'Y' * qubits
    pool.append(all_Y)

    return pool


def mixer_pool_multi(qubits):
    """
    生成双比特的混合器操作池，支持 {XX, YY, YZ, ZY} 类型
    
    生成所有量子比特对的双比特操作：
    - XX01, XX02, ..., XX12, XX13, ...
    - YY01, YY02, ..., YZ01, YZ02, ...
    - ZY01, ZY02, ...
    
    参数:
        qubits: 总量子比特数
    
    返回:
        list: 双比特混合器操作字符串列表
    """
    number_pairs = []
    # 生成所有可能的无序量子比特对 (i,j)，其中 i < j
    for i in range(qubits):
        for j in range(i + 1, qubits):
            number_pairs.append(f'{i}{j}')

    # 定义支持的双比特操作类型
    letter_pairs = ['XX', 'YY', 'YZ', 'ZY']

    pool = []
    for num in number_pairs:
        for let in letter_pairs:
            pool.append(f'{let}{num}')

    return pool


def build_mixer_pool(qubits):
    """
    整合单比特、双比特和全比特混合器池
    
    合并结果：
    - 单比特操作池
    - 双比特操作池
    - 全比特操作池（已在单比特池中包含）
    
    参数:
        qubits: 总量子比特数
    
    返回:
        list: 完整的混合器操作池
    """
    single_pool = mixer_pool_single(qubits)
    multi_pool = mixer_pool_multi(qubits)
    return single_pool + multi_pool


# ---------- 3.4 QAOA线路构建函数 ----------

def qaoa_cost(circ, hamil, gamma, n_qubits_per_node):
    """
    将哈密顿量Hc转换为量子演化层，添加到线路中
    
    支持两种模式：
    1. 参数化模式：gamma为参数名（如"gamma_0"）
    2. 数值模式：gamma为具体数值
    
    参数:
        circ: 目标量子线路对象
        hamil: 目标哈密顿量（包含terms属性）
        gamma: 演化参数（参数名字符串或数值）
        n_qubits_per_node: 每个节点编码的量子比特数
    
    返回:
        Circuit: 添加了演化层的线路
    """
    hamil_terms = hamil.terms.items()

    if isinstance(gamma, str):
        # 参数化场景（gamma为参数名）
        for term, coeff in hamil_terms:
            try:
                # 提取系数实部
                if isinstance(coeff, ParameterResolver):
                    coeff_const = coeff.const if hasattr(coeff, 'const') else 0.0
                    coeff_scalar = float(coeff_const.real)
                else:
                    coeff_scalar = float(coeff.real) if isinstance(coeff, complex) else float(coeff)

                if not np.isfinite(coeff_scalar):
                    raise ValueError(f"无效系数: {coeff_scalar}")

                # 构建参数解析器并添加Rzz门
                pr = ParameterResolver({gamma: coeff_scalar})
                qubits = [q for q, _ in term]
                if len(qubits) == 2:
                    circ += Rzz(pr).on(qubits)
                elif len(qubits) == 1:
                    q = qubits[0]
                    circ += Rzz(pr).on([q, q])

            except Exception as e:
                print(f"处理哈密顿量项 {term} 失败: {e}")
                raise
    else:
        # 数值场景（gamma为具体数值）
        for term, coeff in hamil_terms:
            try:
                coeff_scalar = float(coeff.real) if isinstance(coeff, complex) else float(coeff)
                if not np.isfinite(coeff_scalar):
                    raise ValueError(f"无效系数: {coeff_scalar}")

                qubits = [q for q, _ in term]
                if len(qubits) == 2:
                    circ += Rzz(gamma * coeff_scalar).on(qubits)
                elif len(qubits) == 1:
                    q = qubits[0]
                    circ += Rzz(gamma * coeff_scalar * 0.5).on([q, q])

            except Exception as e:
                print(f"处理哈密顿量项 {term} 失败: {e}")
                raise

    circ.barrier()
    return circ


def qaoa_mixer(circ, mixer_str, beta, qubits):
    """
    根据 mixer 字符串描述，向量子线路添加一层对应的混合层操作
    
    支持的混合器类型：
    - 单比特：X{i}, Y{i}
    - 双比特：XX{ij}, YY{ij}, YZ{ij}, XY{ij}, XZ{ij}
    - 全比特：X...（qubits个X）, Y...（qubits个Y）
    
    参数:
        circ: 目标量子线路对象
        mixer_str: 混合器操作字符串（如 "XX12", "Y3", "X4Y5YZ67" 等）
        beta: 变分参数名或数值
        qubits: 总量子比特数（用于全量子比特操作判断）
    """
    # 提取操作符字母和量子比特编号
    letters = re.findall(r'[A-Z]', mixer_str)
    numbers = re.findall(r'\d+', mixer_str)
    letters_str = ''.join(letters)
    nums = [int(num) for num in numbers]

    pr = ParameterResolver({beta: 2})

    # 处理双量子比特操作
    if len(nums) == 2:
        if letters_str == 'XX':
            circ += Rxx(pr).on(nums)
        elif letters_str == 'YY':
            circ += Ryy(pr).on(nums)
        elif letters_str == 'YZ':
            circ += Ryz(pr).on(nums)
        elif letters_str == 'XY':
            circ += Rxy(pr).on(nums)
        elif letters_str == 'XZ':
            circ += Rxz(pr).on(nums)
    # 处理全量子比特操作
    elif letters_str == 'X' * qubits:
        for i in range(qubits):
            circ += RX(pr).on(i)
    elif letters_str == 'Y' * qubits:
        for i in range(qubits):
            circ += RY(pr).on(i)
    # 处理单量子比特操作
    elif len(nums) == 1:
        if letters_str == 'X':
            circ += RX(pr).on(nums[0])
        elif letters_str == 'Y':
            circ += RY(pr).on(nums[0])

    circ.barrier()


def derivative(hc, qubits, mixer_str, circ, n_qubits_per_node):
    """
    计算对易子期望值 |⟨[Hc, Hm]⟩|，作为选择混合器的梯度
    
    梯度计算：G = |⟨-i[Hc, Hm]⟩|
    
    参数:
        hc: 目标哈密顿量
        qubits: 总量子比特数
        mixer_str: 混合器字符串（如 "XX12"）
        circ: 当前量子线路
        n_qubits_per_node: 每个节点的量子比特数
    
    返回:
        float: 梯度值（绝对值）
    """
    hm = QubitOperator()
    letters = re.findall(r'[A-Z]', mixer_str)
    numbers = re.findall(r'\d+', mixer_str)
    letters_str = ''.join(letters)
    nums = [int(num) for num in numbers]

    if not letters_str or not nums:
        return 0.0

    # 根据混合器类型构建对应的哈密顿量
    if len(nums) == 1:
        q = nums[0]
        if letters_str == 'X':
            hm += QubitOperator(f"X{q}")
        elif letters_str == 'Y':
            hm += QubitOperator(f"Y{q}")

    elif len(nums) == 2:
        q1, q2 = nums
        if letters_str == 'XX':
            hm += QubitOperator(f"X{q1} X{q2}")
        elif letters_str == 'YY':
            hm += QubitOperator(f"Y{q1} Y{q2}")
        elif letters_str == 'XZ':
            hm += QubitOperator(f"X{q1} Z{q2}")
        elif letters_str == 'YZ':
            hm += QubitOperator(f"Y{q1} Z{q2}")
        elif letters_str == 'XY':
            hm += QubitOperator(f"X{q1} Y{q2}")

    elif len(nums) == qubits and letters_str == 'X' * qubits:
        for i in range(qubits):
            hm += QubitOperator(f"X{i}")

    elif len(nums) == qubits and letters_str == 'Y' * qubits:
        for i in range(qubits):
            hm += QubitOperator(f"Y{i}")

    # 计算对易子 [-i, [Hc, Hm]] 的期望值
    try:
        comm = (-1j) * commutator(hc, hm)
        ham_comm = Hamiltonian(comm)

        sim = Simulator("mqvector", qubits)
        expectation = sim.get_expectation(ham_comm, circ)
        return abs(expectation.real)
    except:
        return 0.0


# ==============================================================================
# 4. 着色辅助函数
# ==============================================================================

def count_conflicts(coloring, graph, verbose=False):
    """
    计算着色方案中的冲突数（相邻节点同色的边数量）
    
    参数:
        coloring: 着色方案字典，{节点: 颜色}
        graph: networkx图对象
        verbose: 是否输出详细冲突信息（默认False）
    
    返回:
        int: 冲突边数量（无效输入返回-1）
    """
    # 1. 输入验证
    if not isinstance(graph, nx.Graph):
        print("错误：graph必须是networkx Graph对象")
        return -1

    if not isinstance(coloring, dict):
        print("错误：coloring必须是字典类型")
        return -1

    if not coloring:
        if verbose:
            print("警告：着色方案为空，冲突数记为0")
        return 0

    # 2. 提取有效节点并预处理
    valid_nodes = set(coloring.keys())
    graph_nodes = set(graph.nodes)
    missing_nodes = graph_nodes - valid_nodes

    if missing_nodes and verbose:
        print(f"警告：存在未着色的节点 {missing_nodes}，这些节点的边将被忽略")

    # 3. 冲突检测
    conflicts = 0
    conflicting_edges = []

    for u, v in graph.edges:
        if u in valid_nodes and v in valid_nodes:
            try:
                if coloring[u] == coloring[v]:
                    conflicts += 1
                    conflicting_edges.append((u, v, coloring[u]))
            except KeyError as e:
                if verbose:
                    print(f"警告：节点 {e} 在着色方案中不存在")

    # 4. 输出详细信息
    if verbose:
        print(f"总边数: {graph.number_of_edges()}, 有效着色边数: {graph.number_of_edges() - len(missing_nodes)}")
        print(f"冲突边数: {conflicts}")
        if conflicts > 0:
            print("冲突详情（节点对, 颜色）:", conflicting_edges)

    return conflicts


def extract_coloring(result, graph, k):
    """
    从量子采样结果中提取着色方案
    
    参数:
        result: 量子采样结果对象（包含data属性）
        graph: 待着色的图
        k: 颜色数量
    
    返回:
        dict or None: 着色方案字典 {节点: 颜色}，失败返回None
    """
    try:
        counts = result.data
        if not counts:
            return None

        # 选择出现概率最高的测量结果作为着色方案
        max_count = -1
        best_bitstring = None
        for bitstring, count in counts.items():
            if count > max_count:
                max_count = count
                best_bitstring = bitstring

        # 将比特串转换为颜色分配
        coloring = {}
        for i, node in enumerate(graph.nodes):
            color = int(best_bitstring[i * int(math.log2(k)): (i + 1) * int(math.log2(k))], 2) % k
            coloring[node] = color

        return coloring
    except Exception as e:
        print(f"提取着色方案时出错: {e}")
        return None


def find_conflict_edges(coloring, graph):
    """
    高效检测冲突边
    
    参数:
        coloring: 着色方案字典
        graph: networkx图对象
    
    返回:
        list: 冲突边列表 [(u, v), ...]
    """
    conflict_edges = []
    valid_nodes = set(coloring.keys())
    for u, v in graph.edges():
        if u in valid_nodes and v in valid_nodes and coloring[u] == coloring[v]:
            conflict_edges.append((u, v))
    return conflict_edges


# ---------- 4.1 环图着色专用函数 ----------

def get_cycle_order(cycle_graph):
    """
    获取环图的节点顺序（按环的连接顺序）
    
    参数:
        cycle_graph: 环图对象
    
    返回:
        list: 节点顺序列表
    """
    if not is_cycle_graph(cycle_graph):
        raise ValueError("输入不是有效环图，无法获取环顺序")
    start_node = next(iter(cycle_graph.nodes))
    cycle_order = [start_node]
    current = start_node
    prev = None
    while len(cycle_order) < cycle_graph.number_of_nodes():
        neighbors = [n for n in cycle_graph.neighbors(current) if n != prev]
        if len(neighbors) != 1:
            raise RuntimeError(f"环图结构异常：节点{current}的非前序邻居数={len(neighbors)}")
        next_node = neighbors[0]
        cycle_order.append(next_node)
        prev, current = current, next_node
    return cycle_order


def cycle_graph_coloring(cycle_graph):
    """
    环图专用着色算法
    
    - 偶环：使用2色交替着色
    - 奇环：使用3色循环着色
    
    参数:
        cycle_graph: 环图对象
    
    返回:
        tuple: (coloring, k)
            - coloring: 着色方案字典
            - k: 使用的颜色数（偶环为2，奇环为3）
    """
    n = cycle_graph.number_of_nodes()
    cycle_order = get_cycle_order(cycle_graph)
    coloring = {}
    if n % 2 == 0:
        for idx, node in enumerate(cycle_order):
            coloring[node] = idx % 2
        return coloring, 2
    else:
        for idx, node in enumerate(cycle_order):
            coloring[node] = idx % 3
        return coloring, 3


# ---------- 4.2 贪心着色函数 ----------

def assign_colors_in_order(graph, ordered_nodes, k, vertex_colors=None):
    """
    按指定顺序为图节点分配颜色（贪心策略）
    
    参数:
        graph: 待着色的图
        ordered_nodes: 节点排序列表
        k: 可用颜色数
        vertex_colors: 预设的顶点颜色字典
    
    返回:
        tuple: (coloring, required_k)
            - coloring: 着色方案字典
            - required_k: 实际需要的颜色数
    """
    coloring = {}
    required_k = k
    if vertex_colors and isinstance(vertex_colors, dict):
        valid_precolors = {n: c for n, c in vertex_colors.items() if n in graph.nodes}
        for node, color in valid_precolors.items():
            coloring[node] = color
            required_k = max(required_k, color + 1)
    for node in ordered_nodes:
        if node in coloring:
            continue
        used_colors = set()
        for neighbor in graph.neighbors(node):
            if neighbor in coloring:
                used_colors.add(coloring[neighbor])
        assigned = False
        for color in range(k):
            if color not in used_colors:
                coloring[node] = color
                assigned = True
                break
        if not assigned:
            new_color = max(used_colors) + 1 if used_colors else 0
            coloring[node] = new_color
            required_k = new_color + 1
    return coloring, required_k


def validate_min_k(graph, initial_k):
    """
    验证并调整最小k值，确保满足着色理论下限
    
    参数:
        graph: 待着色的图
        initial_k: 初始k值
    
    返回:
        int: 验证后的最小k值
    """
    n = graph.number_of_nodes()
    if n <= 1:
        return 1
    if is_cycle_graph(graph):
        return 2 if n % 2 == 0 else 3
    if is_complete_graph(graph):
        return max(initial_k, n)
    max_degree = max(graph.degree(node) for node in graph.nodes)
    return max(initial_k, max_degree)


def _greedy_coloring_from_max_degree(graph, k):
    """
    从最大度节点开始进行贪心着色
    
    参数:
        graph: 待着色的图
        k: 可用颜色数
    
    返回:
        dict or None: 着色方案字典，失败返回None
    """
    nodes_by_degree = sorted(graph.nodes, key=lambda x: graph.degree(x), reverse=True)
    coloring = {}

    for node in nodes_by_degree:
        used_colors = set()
        for neighbor in graph.neighbors(node):
            if neighbor in coloring:
                used_colors.add(coloring[neighbor])

        for color in range(k):
            if color not in used_colors:
                coloring[node] = color
                break
        else:
            return None

    return coloring


def _resolve_conflicts_with_greedy(graph, k, max_attempts=100):
    """
    多次尝试贪心着色直到无冲突或达到最大尝试次数
    
    参数:
        graph: 待着色的图
        k: 可用颜色数
        max_attempts: 最大尝试次数
    
    返回:
        tuple: (coloring, conflicts)
            - coloring: 着色方案字典
            - conflicts: 冲突数
    """
    if k <= 0:
        return None, float('inf')

    # 先检查是否为完全图
    if is_complete_graph(graph):
        required_k = len(graph.nodes)
        if k >= required_k:
            coloring = {i: i for i in range(required_k)}
            return coloring, 0
        else:
            return None, float('inf')

    # 多次尝试贪心着色
    for attempt in range(max_attempts):
        coloring = _greedy_coloring_from_max_degree(graph, k)
        if coloring is None:
            continue

        conflicts = count_conflicts(coloring, graph)
        if conflicts == 0:
            return coloring, 0

    # 返回最后一次的着色和冲突数
    final_coloring = _greedy_coloring_from_max_degree(graph, k) or {}
    final_conflicts = count_conflicts(final_coloring, graph) if final_coloring else float('inf')
    return final_coloring, final_conflicts


def get_subgraph_coloring(subgraph, final_coloring, min_k):
    """
    获取子图着色方案，修复颜色范围和节点覆盖问题
    
    参数:
        subgraph: 子图对象
        final_coloring: 全局着色方案
        min_k: 颜色数
    
    返回:
        dict: 子图着色方案字典
    """
    if not subgraph or len(subgraph.nodes) == 0:
        return {}

    if not final_coloring or not isinstance(final_coloring, dict):
        return {node: 0 for node in subgraph.nodes}

    # 确保min_k有效
    min_k = max(1, min_k)

    # 按节点度排序
    nodes_by_degree = sorted(subgraph.nodes, key=lambda x: subgraph.degree(x), reverse=True)

    sub_coloring = {}
    for node in nodes_by_degree:
        if node in final_coloring:
            sub_coloring[node] = final_coloring[node] % min_k
        else:
            neighbors = list(subgraph.neighbors(node))
            used_colors = {sub_coloring[neigh] for neigh in neighbors if neigh in sub_coloring}
            color = 0
            while color in used_colors and color < min_k:
                color += 1
            sub_coloring[node] = color % min_k

    return sub_coloring


# ==============================================================================
# 5. 可视化工具函数
# ==============================================================================

def plot_original_graph(graph, title="Original Graph Visualization", index=None,
                       layout_seed=42, node_size=300, font_size=12,
                       canvas_scale=0.8):
    """
    无着色原始图可视化

    特性：
    - 所有节点均为单一灰色，无着色区分
    - 标签显示节点原始编号
    - 布局与着色图保持一致

    参数:
        graph: 待可视化的图
        title: 图标题
        index: 图索引（未使用）
        layout_seed: 布局随机种子
        node_size: 节点大小
        font_size: 字体大小
        canvas_scale: 画布缩放系数
    """
    if not graph or len(graph.nodes) == 0:
        print("Warning: Invalid or empty graph, cannot visualize")
        return

    # 动态画布适配
    num_nodes = len(graph.nodes)
    fig_width = min(10 + (num_nodes // 10) * 2, 22)
    fig_height = min(8 + (num_nodes // 10) * 1.6, 18)
    plt.figure(figsize=(fig_width, fig_height))

    # 稀疏布局
    pos = nx.spring_layout(
        graph,
        seed=layout_seed,
        scale=canvas_scale,
        k=1.5 / np.sqrt(num_nodes),
        iterations=50
    )

    # 所有节点使用单一灰色
    node_color = '#AAAAAA'
    node_sizes = [node_size for _ in graph.nodes]

    # 分步骤绘制
    nx.draw_networkx_edges(graph, pos, width=2, alpha=0.7, edge_color='#888888')
    nx.draw_networkx_nodes(graph, pos, node_color=node_color, node_size=node_sizes,
                           edgecolors='#333333', linewidths=1.5)
    nx.draw_networkx_labels(graph, pos, labels={node: str(node) for node in graph.nodes()},
                           font_size=font_size, font_family='sans-serif', font_weight='bold')

    isolated_count = sum(1 for node in graph.nodes() if graph.degree(node) == 0)
    plt.title(f"{title}\n(Total Nodes={num_nodes}, Isolated Nodes={isolated_count}, Edges={graph.number_of_edges()})",
              fontsize=16, pad=25)
    plt.axis('off')
    plt.tight_layout(pad=2.0)
    # plt.show()  # 使用非交互式后端，不显示图片
    plt.close()


def plot_New_IDs_subgraphs(subgraphs, sub_mappings, title="Divided Subgraphs"):
    """
    绘制带新编号的子图（使用映射关系显示原始ID）
    
    参数:
        subgraphs: 子图列表
        sub_mappings: 节点映射列表
        title: 主标题
    """
    num_subgraphs = len(subgraphs)
    cols = math.ceil(math.sqrt(num_subgraphs))
    rows = math.ceil(num_subgraphs / cols)
    plt.figure(figsize=(12, 8))

    for i, (sub, mapping) in enumerate(zip(subgraphs, sub_mappings)):
        pos = nx.spring_layout(sub)

        # 检查子图和映射字典是否匹配
        sub_nodes = set(sub.nodes)
        mapping_keys = set(mapping.keys())

        if sub_nodes != mapping_keys:
            mapping = {old: new for new, old in enumerate(sub.nodes())}
            sub_mappings[i] = mapping

        for node in sub_nodes:
            if node not in pos:
                pos[node] = (0, 0)

        plt.subplot(rows, cols, i + 1)
        try:
            nx.draw_networkx(sub, pos, labels=mapping, with_labels=True, node_size=500, width=2,
                           font_size=16, font_family='sans-serif')
        except KeyError as e:
            print(f"Error drawing subgraph {i + 1}: {e}")
            continue
        plt.title(f"Subgraph {i + 1} (New IDs)")

    plt.suptitle(title)
    # plt.show()  # 使用非交互式后端，不显示图片
    plt.close()


def plot_Original_IDs_subgraphs(subgraphs, title="Subgraphs with Original Node IDs"):
    """
    绘制带原始编号的子图
    
    参数:
        subgraphs: 子图列表
        title: 主标题
    """
    num_subgraphs = len(subgraphs)
    cols = math.ceil(math.sqrt(num_subgraphs))
    rows = math.ceil(num_subgraphs / cols)

    plt.figure(figsize=(12, 8))
    for i, subgraph in enumerate(subgraphs):
        pos = nx.spring_layout(subgraph)
        plt.subplot(rows, cols, i + 1)
        nx.draw_networkx(subgraph, pos, with_labels=True, node_size=500, width=2,
                         font_size=16, font_family='sans-serif')
        plt.title(f"Subgraph {i + 1} (Original IDs)")
    plt.suptitle(title)
    # plt.show()  # 使用非交互式后端，不显示图片
    plt.close()


def plot_New_IDs_colored_subgraphs(subgraphs, sub_colorings, sub_mappings, min_k_list=None,
                                   filename=None, output_dir=None, title=None):
    """
    绘制带新编号的着色子图

    参数:
        subgraphs: 子图列表
        sub_colorings: 子图着色方案列表
        sub_mappings: 节点映射列表
        min_k_list: 最小色数列表
        filename: 输出文件名（不含扩展名）
        output_dir: 输出目录
        title: 主标题
    """
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "subgraph_visualizations")
    os.makedirs(output_dir, exist_ok=True)

    if not subgraphs:
        print("Warning: Subgraph list is empty, cannot plot")
        return

    plot_title = title if title else "Subgraph Coloring Results (Renumbered)"

    num_subgraphs = len(subgraphs)
    cols = min(math.ceil(math.sqrt(num_subgraphs)), 4)
    rows = math.ceil(num_subgraphs / cols)

    fig_width = 6 * cols
    fig_height = 5 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten() if num_subgraphs > 1 else [axes]

    # 处理最小色数列表默认值
    if min_k_list is None:
        min_k_list = []
        for coloring in sub_colorings:
            if coloring and isinstance(coloring, dict):
                min_k = len(set(coloring.values()))
                min_k_list.append(max(1, min_k))
            else:
                min_k_list.append(1)
    min_k_list = min_k_list[:num_subgraphs] + [1] * (num_subgraphs - len(min_k_list))

    for i in range(num_subgraphs):
        sub = subgraphs[i] if i < len(subgraphs) else None
        initial_coloring = sub_colorings[i] if i < len(sub_colorings) else None
        mapping = sub_mappings[i] if i < len(sub_mappings) else None
        k = min_k_list[i]

        if not sub or len(sub.nodes) == 0:
            continue

        if k <= 0:
            k = 1
            min_k_list[i] = k

        ax = axes[i] if i < len(axes) else axes[-1]
        ax.clear()

        node_sizes = [max(sub.degree(node) * 100 + 300, 400) for node in sub.nodes]
        cmap = plt.cm.get_cmap('tab20' if k > 10 else 'tab10', max(1, k))

        pos = nx.spring_layout(sub, seed=42, k=0.8 / math.sqrt(sub.number_of_nodes()), iterations=100)

        # 检查冲突并尝试贪心着色
        final_coloring = None
        conflicts = 0

        if initial_coloring and isinstance(initial_coloring, dict):
            conflicts = count_conflicts(initial_coloring, sub)
            if conflicts > 0:
                final_coloring, conflicts = _resolve_conflicts_with_greedy(sub, k)
            else:
                final_coloring = initial_coloring
        else:
            final_coloring, conflicts = _resolve_conflicts_with_greedy(sub, k)

        if not final_coloring or conflicts > 0:
            if not final_coloring:
                final_coloring = {node: 0 for node in sub.nodes}

        # 分步骤绘制
        nx.draw_networkx_edges(sub, pos, width=1.5, alpha=0.8, edge_color='#888888', ax=ax)
        nx.draw_networkx_nodes(sub, pos, node_color=[final_coloring.get(node, 0) for node in sub.nodes],
                             node_size=node_sizes, cmap=cmap, edgecolors='#333333', linewidths=1, ax=ax)
        nx.draw_networkx_labels(sub, pos, labels=mapping, font_size=10, font_family='sans-serif', ax=ax)

        node_count = sub.number_of_nodes()
        edge_count = sub.number_of_edges()
        conflict_status = "Conflict-Free" if conflicts == 0 else f"Conflicts: {conflicts}"
        ax.set_title(f"Subgraph {i + 1}\n(k={k}, Nodes={node_count}, Edges={edge_count}, {conflict_status})", fontsize=10)
        ax.axis('off')

    # 隐藏多余的子图
    for j in range(num_subgraphs, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Subgraph Coloring Results (Renumbered)", y=0.99, fontsize=14)
    plt.tight_layout()

    if filename:
        # 统一命名格式: {filename}_subgraphs_renumbered.pdf
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(output_dir, f"{base_name}_subgraphs_renumbered.pdf")
        try:
            plt.savefig(save_path, dpi=800, bbox_inches='tight', pad_inches=0.5)
            print(f"Subgraph coloring (renumbered) saved: {save_path}")
        except Exception as e:
            print(f"Failed to save subgraph coloring (renumbered): {str(e)}")

    # plt.show()  # 使用非交互式后端，不显示图片
    plt.close(fig)


def plot_Original_IDs_colored_subgraphs(
        subgraphs,
        subgraph_colorings,
        title="Subgraph Coloring (Original IDs)",
        min_k_list=None,
        filename=None,
        output_dir=None
):
    """
    绘制带原始编号的着色子图

    参数:
        subgraphs: 子图列表
        subgraph_colorings: 子图着色方案列表
        title: 主标题
        min_k_list: 最小色数列表
        filename: 输出文件名（不含扩展名）
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "subgraph_visualizations")
    os.makedirs(output_dir, exist_ok=True)

    if not subgraphs:
        print("Warning: Subgraph list is empty, cannot plot")
        return

    num_subgraphs = len(subgraphs)
    cols = min(math.ceil(math.sqrt(num_subgraphs)), 4)
    rows = math.ceil(num_subgraphs / cols)

    fig_width = 6 * cols
    fig_height = 5 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten() if num_subgraphs > 1 else [axes]

    # 处理最小色数列表
    if min_k_list is None:
        min_k_list = [max(1, len(set(coloring.values())))
                      if coloring and isinstance(coloring, dict) else 1
                      for coloring in subgraph_colorings]
    min_k_list = min_k_list[:num_subgraphs] + [1] * (num_subgraphs - len(min_k_list))

    for i in range(num_subgraphs):
        subgraph = subgraphs[i] if i < len(subgraphs) else None
        initial_coloring = subgraph_colorings[i] if i < len(subgraph_colorings) else None
        k = min_k_list[i]

        if not subgraph or len(subgraph.nodes) == 0:
            continue

        if k <= 0:
            k = 1
            min_k_list[i] = k

        ax = axes[i] if i < len(axes) else axes[-1]
        ax.clear()

        node_sizes = [max(subgraph.degree(node) * 100 + 300, 400) for node in subgraph.nodes]
        cmap = plt.cm.get_cmap('tab20' if k > 10 else 'tab10', max(1, k))

        pos = nx.spring_layout(subgraph, seed=42, k=0.8 / math.sqrt(subgraph.number_of_nodes()), iterations=100)

        # 检查冲突并尝试贪心着色
        final_coloring = None
        conflicts = 0

        if initial_coloring and isinstance(initial_coloring, dict):
            conflicts = count_conflicts(initial_coloring, subgraph)
            if conflicts > 0:
                final_coloring, conflicts = _resolve_conflicts_with_greedy(subgraph, k)
            else:
                final_coloring = initial_coloring
        else:
            final_coloring, conflicts = _resolve_conflicts_with_greedy(subgraph, k)

        if not final_coloring or conflicts > 0:
            if not final_coloring:
                final_coloring = {node: 0 for node in subgraph.nodes}

        # 分步骤绘制
        nx.draw_networkx_edges(subgraph, pos, width=1.5, alpha=0.8, edge_color='#888888', ax=ax)
        nx.draw_networkx_nodes(subgraph, pos, node_color=[final_coloring.get(node, 0) for node in subgraph.nodes],
                             node_size=node_sizes, cmap=cmap, edgecolors='#333333', linewidths=1, ax=ax)
        nx.draw_networkx_labels(subgraph, pos, font_size=10, font_family='sans-serif', ax=ax)

        node_count = subgraph.number_of_nodes()
        edge_count = subgraph.number_of_edges()
        conflict_status = "Conflict-Free" if conflicts == 0 else f"Conflicts: {conflicts}"
        ax.set_title(f"Subgraph {i + 1}\n(k={k}, Nodes={node_count}, Edges={edge_count}, {conflict_status})", fontsize=10)
        ax.axis('off')

    # 隐藏多余的子图
    for j in range(num_subgraphs, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title, y=0.99, fontsize=14)
    plt.tight_layout()

    if filename:
        # 统一命名格式: {filename}_subgraphs_original.pdf
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(output_dir, f"{base_name}_subgraphs_original.pdf")
        try:
            plt.savefig(save_path, dpi=800, bbox_inches='tight', pad_inches=0.5)
            print(f"Subgraph coloring (original IDs) saved: {save_path}")
        except Exception as e:
            print(f"Failed to save subgraph coloring (original IDs): {str(e)}")

    # plt.show()  # 使用非交互式后端，不显示图片
    plt.close(fig)


def visualize_graph(graph, coloring=None, title="Graph Visualization", index=None, min_k=None,
                    layout_seed=42, node_size=800, font_size=12, canvas_scale=0.8, filename=None, processing_time=None):
    """
    可视化全局图着色结果

    参数:
        graph: 待可视化的图
        coloring: 着色方案字典
        title: 图标题
        index: 图索引
        min_k: 颜色数
        layout_seed: 布局随机种子
        node_size: 节点大小
        font_size: 字体大小
        canvas_scale: 画布缩放系数
        filename: 输出文件名（不含扩展名）
        processing_time: 执行时间（秒）
    """
    if not graph or len(graph.nodes) == 0:
        print("Warning: Invalid or empty graph, cannot visualize")
        return

    save_dir = os.path.join(BASE_DIR, "graph_visualizations")
    os.makedirs(save_dir, exist_ok=True)

    # 动态画布大小
    num_nodes = len(graph.nodes)
    fig_width = min(10 + (num_nodes // 10) * 2, 20)
    fig_height = min(8 + (num_nodes // 10) * 1.6, 16)
    plt.figure(figsize=(fig_width, fig_height))

    # 稀疏布局
    pos = nx.spring_layout(
        graph,
        seed=layout_seed,
        scale=canvas_scale,
        k=1.2 / np.sqrt(num_nodes),
        iterations=50
    )

    # 着色方案处理
    if coloring is None or not isinstance(coloring, dict):
        coloring = {node: 0 for node in graph.nodes}
        min_k = 1
    else:
        missing_nodes = [node for node in graph.nodes if node not in coloring]
        if missing_nodes:
            used_colors = set(coloring.values())
            new_color = 0
            while new_color in used_colors:
                new_color += 1
            for node in missing_nodes:
                coloring[node] = new_color
        if min_k is None:
            min_k = len(set(coloring.values()))
        min_k = max(1, min_k)
        coloring = {node: color % min_k for node, color in coloring.items()}

    node_sizes = [node_size for _ in graph.nodes]
    cmap = plt.cm.get_cmap('tab20' if min_k > 10 else 'tab10', min_k)

    # 分步骤绘制
    nx.draw_networkx_edges(graph, pos, width=2, alpha=1, edge_color='#888888')
    node_collection = nx.draw_networkx_nodes(graph, pos, node_color=[coloring.get(node, 0) for node in graph.nodes()],
                                          node_size=node_sizes, cmap=cmap, edgecolors='#333333', linewidths=1.5)
    label_dict = nx.draw_networkx_labels(graph, pos, labels={node: str(node) for node in graph.nodes()},
                                        font_size=font_size, font_family='sans-serif', font_weight='bold')

    isolated_count = sum(1 for node in graph.nodes() if graph.degree(node) == 0)
    edge_count = graph.number_of_edges()

    # 构建标题，包含执行时间信息
    title_parts = [title]
    title_parts.append(f"(k={min_k}, Nodes={num_nodes}, Isolated={isolated_count}, Edges={edge_count})")
    if processing_time is not None:
        title_parts.append(f"Time={processing_time:.2f}s")

    full_title = "\n".join(title_parts)
    plt.title(full_title, fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout(pad=2.0)

    if index is not None:
        try:
            # 统一命名格式: {filename}_final_coloring_k.pdf
            base_name = os.path.splitext(filename)[0] if filename else "".join([c for c in title if c.isalnum() or c in " _-"]).strip().replace(" ", "_")

            # file_name = f"{base_name}_final_coloring_{min_k}_n{num_nodes}_e{edge_count}.pdf"
            file_name = f"{base_name}_final_coloring_{min_k}.pdf"
            file_name = file_name.replace("__", "_").replace("/", "_")

            save_path = os.path.join(save_dir, file_name)
            plt.savefig(save_path, dpi=800, bbox_inches='tight', pad_inches=0.6)
            print(f"Final coloring saved: {save_path}")
        except Exception as e:
            print(f"保存图片失败: {str(e)}")

    # plt.show()  # 使用非交互式后端，不显示图片
    plt.close()


# ==============================================================================
# 6. 异常处理函数
# ==============================================================================

def handle_exception(func_name, index, e):
    """
    统一异常处理函数
    
    参数:
        func_name: 函数名称
        index: 图索引
        e: 异常对象
    """
    print(f"Error in {func_name} for graph {index}: {e}")
    traceback.print_exc()