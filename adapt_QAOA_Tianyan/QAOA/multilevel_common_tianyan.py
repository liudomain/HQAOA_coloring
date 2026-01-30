"""本模块提供多层次QAOA图着色算法的共享函数，主要支持：
- 标准QAOA (Standard QAOA): 使用预训练参数
- 天衍平台适配：使用cqlib库，支持真机执行

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
import logging
import csv
from math import log2, ceil
import math
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter

# cqlib 导入
from cqlib.circuits import Circuit, Parameter
from cqlib import TianYanPlatform, QuantumLanguage
from cqlib.utils import LaboratoryUtils
import numpy as np
import networkx as nx
import matplotlib
# 使用非交互式后端，图片显示后不阻塞程序继续执行
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# 统一日志目录路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# numpy 类型转换补丁
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

    logger_name = f"subgraph_processor_{dataset}_{graph_id}"
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 抑制天衍平台 SDK 的等待日志
        tianyan_logger = logging.getLogger('cqlib')
        tianyan_logger.setLevel(logging.WARNING)

    return logger


# ==============================================================================
# 2. 图处理工具函数
# ==============================================================================

def detect_sparse_graph_strategy(graph):
    """
    检测图的稀疏程度并返回相应的处理策略

    参数:
        graph: NetworkX 图对象

    返回:
        tuple: (策略类型, 策略描述, 边密度)
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    # 计算边密度
    if n > 1:
        edge_density = 2 * m / (n * (n - 1))
    else:
        edge_density = 0

    # 根据边密度划分稀疏程度
    if edge_density < 0.05:
        return "ultra_sparse", "极稀疏图，适度划分", edge_density
    elif edge_density < 0.1:
        return "very_sparse", "极度稀疏图，适度划分", edge_density
    elif edge_density < 0.3:
        return "sparse", "稀疏图，适度划分", edge_density
    else:
        return "dense", "密集图，标准划分", edge_density


def calculate_optimal_subgraph_params(graph, max_qubits, max_k, original_num_subgraphs, original_max_nodes):
    """
    根据量子比特限制和图的稀疏程度计算最优的子图划分参数

    参数:
        graph: NetworkX 图对象
        max_qubits: 量子比特数限制
        max_k: 最大颜色数
        original_num_subgraphs: 原始子图数量
        original_max_nodes: 原始最大节点数

    返回:
        tuple: (调整后的子图数量, 调整后的最大节点数, 策略信息)
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    # 检测稀疏程度
    strategy, description, edge_density = detect_sparse_graph_strategy(graph)

    # 计算每个节点需要的量子比特数
    qubits_per_node = math.ceil(math.log2(max_k)) if max_k > 2 else 1

    # 计算理论最大子图节点数
    max_nodes_theory = max_qubits // qubits_per_node if qubits_per_node > 0 else original_max_nodes

    # 根据稀疏程度调整参数
    if strategy == "ultra_sparse":
        # 极稀疏图：不进行子图划分，直接处理
        adjusted_num_subgraphs = 2
        adjusted_max_nodes = min(max_nodes_theory, n)
        strategy_info = f"极稀疏图(密度={edge_density:.4f})，适度划分"

    elif strategy == "very_sparse":
        # 极度稀疏图：使用适度的子图，避免贪心着色
        adjusted_max_nodes = min(max_nodes_theory, max(25, original_max_nodes * 2))
        adjusted_num_subgraphs = max(2, min(original_num_subgraphs, n // adjusted_max_nodes))
        strategy_info = f"极度稀疏图(密度={edge_density:.4f})，适度划分"

    elif strategy == "sparse":
        # 稀疏图：适度调整
        adjusted_max_nodes = min(max_nodes_theory, max(20, int(original_max_nodes * 1.5)))
        adjusted_num_subgraphs = max(2, min(int(original_num_subgraphs * 0.7), n // adjusted_max_nodes))
        strategy_info = f"稀疏图(密度={edge_density:.4f})，适度划分"

    else:
        # 密集图：使用原始策略，但考虑量子比特限制
        adjusted_max_nodes = min(max_nodes_theory, original_max_nodes)
        adjusted_num_subgraphs = original_num_subgraphs
        strategy_info = f"密集图(密度={edge_density:.4f})，标准划分"

    # 确保参数合理性
    adjusted_num_subgraphs = max(1, min(adjusted_num_subgraphs, n))
    adjusted_max_nodes = max(1, adjusted_max_nodes)

    return adjusted_num_subgraphs, adjusted_max_nodes, strategy_info


def divide_graph(graph, num_subgraphs, Q=20, max_nodes=10, max_qubits=None, max_k=None):
    """
    使用 METIS 划分图，并对大子图递归二分直至节点数 ≤ max_nodes

    参数:
        graph: 待划分的图
        num_subgraphs: 目标子图数量
        Q: 图划分平衡因子（默认20）
        max_nodes: 子图最大节点数限制（默认10）
        max_qubits: 量子比特数限制（可选，用于兼容性）
        max_k: 最大颜色数（可选，用于兼容性）

    返回:
        tuple: (subgraphs, mappings)
    """
    # 如果子图数量为1或小于等于节点数，直接返回整个图作为一个子图
    if num_subgraphs == 1:
        mapping = {node: i for i, node in enumerate(graph.nodes)}
        return [graph], [mapping], {}

    if num_subgraphs >= len(graph.nodes):
        subgraphs = [graph.subgraph([node]) for node in graph.nodes]
        mappings = [{node: 0} for node in graph.nodes]
        while len(subgraphs) < num_subgraphs:
            subgraphs.append(nx.Graph())
            mappings.append({})
        return subgraphs, mappings, {}

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

    # 递归二分大子图
    final_subgraphs = []
    final_mappings = []

    def split_until_small(g, depth=0, max_depth=100):
        """递归分割图直到节点数 <= max_nodes

        Args:
            g: NetworkX 图
            depth: 当前递归深度
            max_depth: 最大递归深度（防止无限递归）
        """
        # 检查递归深度
        if depth > max_depth:
            print(f"警告：达到最大递归深度 {max_depth}，当前图节点数：{g.number_of_nodes()}，直接加入子图列表")
            final_subgraphs.append(g)
            final_mappings.append({old: new for new, old in enumerate(g.nodes)})
            return

        # 检查是否满足节点数要求
        if g.number_of_nodes() <= max_nodes:
            final_subgraphs.append(g)
            final_mappings.append({old: new for new, old in enumerate(g.nodes)})
            return

        # 尝试分割图
        try:
            if g.number_of_nodes() >= 2:
                # 使用 METIS 进行二分
                _, part2 = metis.part_graph(g, 2, objtype="cut", ufactor=200)
                nodes0 = [n for n, p in zip(g.nodes, part2) if p == 0]
                nodes1 = [n for n, p in zip(g.nodes, part2) if p == 1]

                # 检查分割结果是否有效
                if len(nodes0) == 0 or len(nodes1) == 0:
                    # METIS分割失败，改用简单的节点列表二分
                    nodes_list = sorted(list(g.nodes))
                    mid = len(nodes_list) // 2
                    nodes0 = nodes_list[:mid]
                    nodes1 = nodes_list[mid:]

                    # 如果仍然有一边为空（极端情况），直接加入子图列表
                    if len(nodes0) == 0 or len(nodes1) == 0:
                        final_subgraphs.append(g)
                        final_mappings.append({old: new for new, old in enumerate(g.nodes)})
                        return

                # 递归处理两个子图
                split_until_small(g.subgraph(nodes0).copy(), depth + 1, max_depth)
                split_until_small(g.subgraph(nodes1).copy(), depth + 1, max_depth)
            else:
                final_subgraphs.append(g)
                final_mappings.append({old: new for new, old in enumerate(g.nodes)})

        except Exception as e:
            print(f"警告：图分割过程中发生错误：{e}，节点数：{g.number_of_nodes()}，深度：{depth}")
            # 发生错误时直接加入子图列表
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

    while len(final_subgraphs) < num_subgraphs:
        final_subgraphs.append(nx.Graph())
        final_mappings.append({})

    return final_subgraphs, final_mappings, {}


def is_complete_graph(graph):
    """判断图是否为完全图"""
    n = graph.number_of_nodes()
    if n <= 1:
        return True
    expected_edges = n * (n - 1) // 2
    return graph.number_of_edges() == expected_edges


def is_odd_cycle(graph):
    """判断图是否为奇环"""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    if n != m:
        return False
    if any(graph.degree(node) != 2 for node in graph.nodes):
        return False
    return n % 2 == 1


def is_cycle_graph(graph):
    """判断图是否为环图"""
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


def is_chain_graph(graph):
    """判断图是否为链式图（路径图）"""
    n = graph.number_of_nodes()
    if n < 2:
        return False
    # 链式图：边数 = 节点数 - 1，且所有节点度数 ≤ 2，且连通
    if graph.number_of_edges() != n - 1:
        return False
    if not all(graph.degree(node) <= 2 for node in graph.nodes):
        return False
    if not nx.is_connected(graph):
        return False
    return True


def chain_graph_coloring(graph):
    """
    链式图着色（使用2种颜色）
    
    Args:
        graph: NetworkX 图对象
        
    Returns:
        tuple: (着色字典, 颜色数k)
    """
    # 获取链式图的起点（度数为1的节点）
    start_nodes = [node for node in graph.nodes if graph.degree(node) == 1]
    
    if not start_nodes:
        # 单个节点的情况
        return {list(graph.nodes)[0]: 0}, 1
    
    # 从起点开始着色
    start_node = start_nodes[0]
    coloring = {start_node: 0}
    
    # BFS遍历链式图，交替着色
    visited = {start_node}
    queue = [start_node]
    
    while queue:
        current = queue.pop(0)
        current_color = coloring[current]
        
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                # 使用与当前节点不同的颜色
                coloring[neighbor] = 1 - current_color
                visited.add(neighbor)
                queue.append(neighbor)
    
    return coloring, 2


def get_graph_signature(graph):
    """生成图的唯一签名，用于检测同构子图"""
    nodes = sorted(graph.nodes, key=lambda x: (graph.degree(x), x))
    node_mapping = {old: i for i, old in enumerate(nodes)}
    normalized_graph = nx.relabel_nodes(graph, node_mapping)

    nodes_sorted = sorted(normalized_graph.nodes)
    edges_sorted = tuple(sorted((u, v) for u, v in normalized_graph.edges))
    degree_sequence = tuple(sorted(normalized_graph.degree(n) for n in nodes_sorted))

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

class RzzGate:
    """组合门: Rzz(theta) = exp(-i*theta/2 * Z⊗Z)"""
    def __init__(self, theta, q1, q2):
        self.theta = theta
        self.q1 = q1
        self.q2 = q2

    def apply(self, circuit):
        decompose_rzz(circuit, self.theta, self.q1, self.q2)


def _add_rzz(self, theta, q1, q2):
    """为 Circuit 添加 Rzz 门方法"""
    RzzGate(theta, q1, q2).apply(self)


def assign_parameters(self, **kwargs):
    """分配参数到线路"""
    if not (hasattr(self, 'parameters') and self.parameters) or not kwargs:
        return self

    qcis_list = list(self.qcis)
    param_map = {}
    for k, v in kwargs.items():
        if isinstance(v, float):
            param_map[str(k)] = f"{v:.8f}".rstrip('0').rstrip('.') if '.' in f"{v:.8f}" else str(int(v))
        else:
            param_map[str(k)] = str(v)

    new_circuit = Circuit(qubits=list(range(self.num_qubits)))

    gate_handlers = {
        'H': lambda c, qs: c.h(qs[0]) if qs else None,
        'X': lambda c, qs: c.x(qs[0]) if qs else None,
        'Y': lambda c, qs: c.y(qs[0]) if qs else None,
        'Z': lambda c, qs: c.z(qs[0]) if qs else None,
        'RX': lambda c, qs, ps: c.rx(qs[0], ps[0]) if qs and ps else None,
        'RY': lambda c, qs, ps: c.ry(qs[0], ps[0]) if qs and ps else None,
        'RZ': lambda c, qs, ps: c.rz(qs[0], ps[0]) if qs and ps else None,
        'CZ': lambda c, qs: c.cz(qs[0], qs[1]) if len(qs) >= 2 else None,
        'B': lambda c, qs: c.barrier(*qs) if qs else None,
        # 天衍平台不支持单独的 M 门，需要在电路最后统一调用 measure_all()
        'M': lambda c, qs: None  # 忽略 M 门，使用 measure_all() 统一测量
    }

    for qcis in qcis_list:
        qcis_str = str(qcis)

        for param_name in sorted(param_map.keys(), key=len, reverse=True):
            qcis_str = qcis_str.replace(param_name, param_map[param_name])

        parts = qcis_str.split()
        if len(parts) >= 2:
            gate_name = parts[0].upper()
            qubits = []
            params = []

            for part in parts[1:]:
                part_upper = part.upper()
                if part_upper.startswith('Q'):
                    try:
                        qubits.append(int(part_upper[1:]))
                    except (ValueError, IndexError):
                        pass
                else:
                    try:
                        params.append(float(part))
                    except ValueError:
                        params.append(part)

            handler = gate_handlers.get(gate_name)
            if handler:
                if gate_name in ['RX', 'RY', 'RZ']:
                    handler(new_circuit, qubits, params)
                else:
                    handler(new_circuit, qubits)

    return new_circuit


Circuit.rzz = _add_rzz
Circuit.assign_parameters = assign_parameters


def decompose_cx(circuit, control, target):
    """将 CX(CNOT) 门分解为 H+CZ+H"""
    circuit.h(target)
    circuit.cz(control, target)
    circuit.h(target)


def decompose_rzz(circuit, theta, q1, q2):
    """分解 Rzz(theta) 门为 cqlib 支持的基本门"""
    decompose_cx(circuit, q2, q1)
    circuit.rz(q2, theta)
    decompose_cx(circuit, q2, q1)
    circuit.rz(q1, theta)


def remap_qubits_in_qcis(qcis_input, qubit_mapping):
    """
    将QCIS字符串中的量子比特索引从逻辑索引映射到物理索引
    同时过滤掉 M 门（天衍平台不支持单独的 M 门）

    参数:
        qcis_input: QCIS指令（字符串或QCIS对象列表）
        qubit_mapping: 量子比特映射字典 {old_qubit: new_qubit}

    返回:
        映射后的QCIS字符串（包含所有门）
    """
    # 处理 QCIS 对象列表或字符串
    if isinstance(qcis_input, list):
        qcis_str = '\n'.join([str(q) for q in qcis_input])
    else:
        qcis_str = str(qcis_input)

    lines = qcis_str.split('\n')
    mapped_lines = []

    for line in lines:
        if not line.strip() or line.strip().startswith('#') or line.strip().startswith('c '):
            # 跳过空行和注释行，不添加到结果中
            continue

        # 提取指令部分
        parts = line.strip().split()
        if not parts:
            continue

        gate_name = parts[0].upper()

        # 不再过滤掉 M 门（天衍平台需要测量门）

        # 处理量子比特参数
        new_parts = [gate_name]
        for part in parts[1:]:
            part_upper = part.upper()
            if part_upper.startswith('Q'):
                try:
                    old_qubit = int(part_upper[1:])
                    if old_qubit in qubit_mapping:
                        new_qubit = qubit_mapping[old_qubit]
                        new_parts.append(f'Q{new_qubit}')
                    else:
                        new_parts.append(part)
                except (ValueError, IndexError):
                    new_parts.append(part)
            else:
                new_parts.append(part)

        mapped_lines.append(' '.join(new_parts))


    result = '\n'.join(mapped_lines)

    # 调试：检查结果是否为空
    if not result.strip():
        print("警告：过滤后的QCIS字符串为空！原始QCIS：")
        print(qcis_str[:200] if len(qcis_str) > 200 else qcis_str)  # 打印前200字符
        print(f"映射字典: {qubit_mapping}")
        return None  # 返回 None 而不是空字符串，便于调用方检测

    return result



def build_hamiltonian_circuit_tianyan(graph, k, gamma_param, vertex_colors=None, nodes_to_recolor=None):
    """QAOA哈密顿量演化线路构建函数"""
    if k < 1:
        raise ValueError(f"颜色数量 k 必须大于0，当前值: {k}")

    if k == 1:
        n_qubits_per_node = 1
    elif k == 2:
        n_qubits_per_node = 1
    else:
        n_qubits_per_node = math.ceil(math.log2(k))

    if nodes_to_recolor is None:
        nodes_to_recolor = list(graph.nodes)
    nodes_to_recolor = [int(node) for node in nodes_to_recolor if node in graph.nodes]

    if not nodes_to_recolor:
        raise ValueError("nodes_to_recolor中无有效节点")

    num_qubits = len(graph.nodes) * n_qubits_per_node
    circuit = Circuit(qubits=list(range(num_qubits)))

    for u, v in graph.edges:
        u, v = int(u), int(v)
        if u not in nodes_to_recolor or v not in nodes_to_recolor:
            continue

        for i in range(n_qubits_per_node):
            qubit1 = u * n_qubits_per_node + i
            qubit2 = v * n_qubits_per_node + i
            decompose_rzz(circuit, gamma_param, qubit1, qubit2)

    return circuit


def qaoa_ansatz_tianyan(graph, k, p=1, trained_params=None, vertex_colors=None,
                         nodes_to_recolor=None, verbose=False, max_qubits=200):
    """QAOA线路生成器 (天衍平台版本)"""
    if k < 1:
        raise ValueError(f"颜色数量 k 必须大于0，当前值: {k}")
    if k == 1:
        n_qubits_per_node = 1
    elif k == 2:
        n_qubits_per_node = 1
    else:
        n_qubits_per_node = math.ceil(math.log2(k))
    num_qubits = len(graph.nodes) * n_qubits_per_node

    # 检查量子比特数是否超过平台限制
    if num_qubits > max_qubits:
        raise ValueError(
            f"量子比特数 {num_qubits} 超过平台限制 {max_qubits}。"
            f"节点数: {len(graph.nodes)}, k={k}, 比特/节点: {n_qubits_per_node}"
        )

    if trained_params:
        gamma_values = [trained_params.get(f'gamma_{layer}', 0.5) for layer in range(p)]
        beta_values = [trained_params.get(f'beta_{layer}', 0.3) for layer in range(p)]
    else:
        gamma_values = [0.5] * p
        beta_values = [0.3] * p

    if verbose:
        print(f"[QAOA] 图节点数: {len(graph.nodes)}, 边数: {graph.number_of_edges()}")
        print(f"[QAOA] k={k}, p={p}, 量子比特数: {num_qubits}")

    circuit = Circuit(qubits=list(range(num_qubits)))

    for qubit in range(num_qubits):
        circuit.h(qubit)
    circuit.barrier(*list(range(num_qubits)))

    for layer in range(p):
        gamma = gamma_values[layer]
        beta = beta_values[layer]

        if nodes_to_recolor is None:
            layer_nodes = list(graph.nodes)
        else:
            layer_nodes = [int(node) for node in nodes_to_recolor if node in graph.nodes]

        for u, v in graph.edges:
            u, v = int(u), int(v)
            if u not in layer_nodes or v not in layer_nodes:
                continue

            for i in range(n_qubits_per_node):
                qubit1 = u * n_qubits_per_node + i
                qubit2 = v * n_qubits_per_node + i
                decompose_rzz(circuit, gamma, qubit1, qubit2)

        for u in graph.nodes:
            for i in range(n_qubits_per_node):
                qubit = int(u * n_qubits_per_node + i)
                circuit.rx(qubit, beta)
        circuit.barrier(*list(range(num_qubits)))

        if verbose:
            print(f"[QAOA] 层 {layer} 完成 (gamma={gamma:.4f}, beta={beta:.4f})")

    return circuit


def extract_coloring_tianyan(result, graph, k, verbose=False, max_all_zeros_ratio=0.8):
    """从天衍真机结果中提取着色方案

    参数:
        result: 天衍平台返回的结果
        graph: 待着色的图
        k: 颜色数量
        verbose: 是否输出详细信息
        max_all_zeros_ratio: 全0态最大允许比例，超过此值仅警告但继续提取（默认0.8）
    """
    try:
        if 'resultStatus' not in result:
            if verbose:
                print("警告：结果中没有 resultStatus 字段")
            return None

        measurement_data = result['resultStatus']
        if not measurement_data or len(measurement_data) == 0:
            if verbose:
                print("警告：测量数据为空")
            return None

        measured_qubits = measurement_data[0]
        shots_data = measurement_data[1:]

        if not shots_data:
            if verbose:
                print("警告：没有采样数据")
            return None

        shots_data_str = []
        for shot in shots_data:
            if isinstance(shot, (list, tuple)):
                shot_str = ''.join(map(str, shot))
            else:
                shot_str = str(shot)
            shots_data_str.append(shot_str)

        # 计算全0态比例
        total_shots = len(shots_data_str)
        all_zeros_count = shots_data_str.count('0' * len(shots_data_str[0]) if shots_data_str else '0')
        all_zeros_ratio = all_zeros_count / total_shots if total_shots > 0 else 1.0

        # 如果全0态比例过高，发出警告但继续尝试提取次优解
        if all_zeros_ratio > max_all_zeros_ratio:
            if verbose:
                print(f"⚠️ 警告: 测量结果中全0态比例较高 ({all_zeros_ratio:.1%})，超过阈值 {max_all_zeros_ratio:.1%}，将尝试从剩余结果中提取次优解")

        # 统计所有非全0的采样结果
        non_zero_shots = [shot for shot in shots_data_str if shot != '0' * len(shots_data_str[0])]
        if verbose:
            print(f"总采样次数: {total_shots}, 全0态次数: {all_zeros_count}, 非全0态次数: {len(non_zero_shots)}")

        # 使用所有采样结果进行统计（包括全0态），但优先选择非全0态
        # 这样即使全0态比例很高，也能从其他概率状态中找到次优解
        bitstring_counts = Counter(shots_data_str)

        if k == 1:
            n_qubits_per_node = 1
        elif k == 2:
            n_qubits_per_node = 1
        else:
            n_qubits_per_node = math.ceil(math.log2(k))

        top_candidates = bitstring_counts.most_common(20)

        best_coloring = None
        best_conflicts = float('inf')
        best_bitstring = None

        if verbose:
            print(f"Top 5 采样状态及冲突数:")
            for i, (bitstring, count) in enumerate(bitstring_counts.most_common(5), 1):
                # 计算此bitstring的冲突数
                temp_coloring = {}
                for j, node in enumerate(graph.nodes):
                    start_idx = j * n_qubits_per_node
                    end_idx = start_idx + n_qubits_per_node

                    if start_idx >= len(bitstring):
                        color = j % k
                    else:
                        if end_idx > len(bitstring):
                            node_bits = bitstring[start_idx:]
                        else:
                            node_bits = bitstring[start_idx:end_idx]

                        if node_bits:
                            try:
                                color = int(node_bits, 2) % k
                            except ValueError:
                                color = j % k
                        else:
                            color = j % k
                    temp_coloring[node] = color
                conflicts = count_conflicts(temp_coloring, graph, verbose=False)
                print(f"  {i}. 状态={bitstring}, 次数={count}, 冲突数={conflicts}")

        for bitstring, count in top_candidates:
            temp_coloring = {}
            for i, node in enumerate(graph.nodes):
                start_idx = i * n_qubits_per_node
                end_idx = start_idx + n_qubits_per_node

                if start_idx >= len(bitstring):
                    color = i % k
                else:
                    if end_idx > len(bitstring):
                        node_bits = bitstring[start_idx:]
                    else:
                        node_bits = bitstring[start_idx:end_idx]

                    if node_bits:
                        try:
                            color = int(node_bits, 2) % k
                        except ValueError:
                            color = i % k
                    else:
                        color = i % k

                temp_coloring[node] = color

            current_conflicts = count_conflicts(temp_coloring, graph, verbose=False)

            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best_coloring = temp_coloring.copy()
                best_bitstring = bitstring

                if verbose:
                    print(f"  找到更好着色: 冲突数={current_conflicts}, 状态={bitstring}")

                if current_conflicts == 0:
                    break

        if verbose:
            print(f"最优着色: 冲突数={best_conflicts}, 状态={best_bitstring}")

        return best_coloring
    except Exception as e:
        print(f"提取着色方案时出错: {e}")
        traceback.print_exc()
        return None


def solve_k_coloring_tianyan_with_training(
    platform, graph, k, p=1, prev_params=None,
    lab_id=None, num_shots=5000, num_steps=50,
    learning_rate=0.01, early_stop_threshold=2, train_num_shots=100,
    algorithm='standard', vertex_colors=None, nodes_to_recolor=None,
    verbose=False, logger=None, graph_name=None,
    stop_on_zero_conflict=False, max_qubits=200
):
    """
    在天衍真机上训练并运行QAOA求解k着色问题

    参数:
        platform: TianYanPlatform 实例
        graph: 待着色的图
        k: 颜色数量
        p: QAOA层数
        prev_params: 热启动参数（上一轮k值的最优参数）
        lab_id: 实验室ID
        num_shots: 最终采样次数
        num_steps: 训练最大迭代次数
        learning_rate: 学习率
        early_stop_threshold: 早停阈值
        train_num_shots: 训练时的采样次数
        algorithm: 算法类型
        vertex_colors: 预设顶点颜色
        nodes_to_recolor: 需要重新着色的节点
        verbose: 是否输出详细信息
        logger: 日志记录器
        graph_name: 数据图名称（用于天衍实验命名）
        stop_on_zero_conflict: 找到无冲突解后是否立即停止训练
        max_qubits: 量子比特数限制（根据机时包限制调整）

    返回:
        tuple: (best_k, best_coloring, conflict_count, query_id, training_result)
    """
    try:
        from datetime import datetime
        import sys
        import os

        # 验证量子比特约束（避免提交超限任务到天衍平台）
        n = graph.number_of_nodes()
        required_qubits = n * ceil(log2(k))
        max_allowed_qubits = 36  # 天衍平台36q机器限制

        if required_qubits > max_allowed_qubits:
            error_msg = (f"子图需要{required_qubits}比特（n={n}, k={k}），"
                       f"超过最大限制{max_allowed_qubits}比特，无法提交到天衍平台")
            if logger:
                logger.error(error_msg)
            if verbose:
                print(f"❌ {error_msg}")

            # 返回贪心着色作为后备
            greedy_coloring = {node: i % k for i, node in enumerate(sorted(graph.nodes()))}
            conflicts = count_conflicts(greedy_coloring, graph)

            training_result_dict = {
                'best_energy': conflicts,
                'best_params': {},
                'energy_history': [],
                'iterations': 0,
                'converged': conflicts == 0,
                'elapsed_time': 0,
                'training_mode': 'qubit_limit_exceeded_fallback'
            }

            return k, greedy_coloring, conflicts, None, training_result_dict

        # 智能判断：极度稀疏或完全图直接使用贪心着色
        EDGE_DENSITY_THRESHOLD = 0.1  # 边密度阈值
        m = graph.number_of_edges()

        # 检查是否为完全图（优先使用最大度判断）
        if n <= 1:
            is_complete = (m == 0)
        else:
            # 方法1: 最大度判断（优先，更高效）
            max_degree = max([d for _, d in graph.degree()]) if n > 0 else 0
            is_complete_by_degree = (max_degree == n - 1)

            # 方法2: 边数判断（备用验证）
            is_complete_by_edges = (m == n * (n - 1) // 2)

            # 两种方法任一满足即判定为完全图
            is_complete = is_complete_by_degree or is_complete_by_edges

            if is_complete_by_degree and not is_complete_by_edges:
                if verbose or logger:
                    msg = f"注意：最大度判断为完全图但边数不匹配 (max_deg={max_degree}, n={n}, m={m})"
                    if logger:
                        logger.warning(msg)
                    if verbose:
                        print(msg)

        # 对于完全图，直接使用贪心着色（无论k值），节省量子资源
        if is_complete:
            if verbose or logger:
                msg = f"检测到完全图 (n={n}, m={m}, max_degree={max_degree})，使用贪心着色节省量子资源"
                if logger:
                    logger.info(msg)
                if verbose:
                    print(msg)

            if k >= n:
                # k >= n 时直接返回完美着色（每个节点使用唯一颜色）
                # 按节点顺序分配颜色：{node: idx}
                perfect_coloring = {node: idx for idx, node in enumerate(sorted(graph.nodes()))}
                if verbose or logger:
                    msg = f"  k={k} >= n={n}，返回完美着色: {perfect_coloring}"
                    if logger:
                        logger.info(msg)
                    if verbose:
                        print(msg)
                training_result_dict = {
                    'best_energy': 0,
                    'best_params': {},
                    'energy_history': [],
                    'iterations': 0,
                    'converged': True,
                    'elapsed_time': 0,
                    'training_mode': 'complete_graph_perfect'
                }
                return k, perfect_coloring, 0, None, training_result_dict
            else:
                # k < n 时使用贪心着色尝试找到最优解
                greedy_coloring = _greedy_coloring_from_max_degree(graph, k)
                conflicts = count_conflicts(greedy_coloring, graph)
                training_result_dict = {
                    'best_energy': conflicts,
                    'best_params': {},
                    'energy_history': [],
                    'iterations': 0,
                    'converged': conflicts == 0,
                    'elapsed_time': 0,
                    'training_mode': 'complete_graph_greedy'
                }
                return k, greedy_coloring, conflicts, None, training_result_dict

        # 移除稀疏图的贪心着色逻辑，强制使用QAOA进行着色
        # 原代码：对边密度 < 0.1 的图直接使用贪心着色，导致着色质量差
        # 修改：所有子图都使用QAOA着色，确保着色质量

        train_params_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'delet', 'adapt_QAOA_Tianyan', 'QAOA')
        if train_params_path not in sys.path:
            sys.path.insert(0, train_params_path)

        try:
            from train_params_tianyan import QAOAParamOptimizer
        except ImportError:
            if verbose or logger:
                error_msg = "train_params_tianyan 不可用，无法进行参数训练"
                if logger:
                    logger.error(error_msg)
                if verbose:
                    print(error_msg)
            training_result_dict = {
                'best_energy': float('inf'),
                'best_params': {},
                'energy_history': [],
                'iterations': 0,
                'converged': False,
                'elapsed_time': 0,
                'training_mode': 'import_error'
            }
            return k, {}, float('inf'), None, training_result_dict

        if verbose:
            print(f"\n{'='*60}")
            print(f"天衍真机QAOA求解（带训练）- k={k}, p={p}")
            print(f"图: 节点数={len(graph.nodes)}, 边数={graph.number_of_edges()}")
            print(f"训练步数: {num_steps}, 学习率: {learning_rate}")
            print(f"早停策略: {'已启用' if stop_on_zero_conflict else '已禁用'}")
            print(f"{'='*60}")

        start_time = time.time()

        optimizer = QAOAParamOptimizer(
            platform=platform,
            graph=graph,
            k=k,
            p=p,
            algorithm=algorithm,
            lab_id=lab_id,
            num_shots=train_num_shots
        )

        if prev_params is not None:
            # 改进热启动参数处理：即使k值变化，也尝试使用可用的参数
            adjusted_params = []
            for layer in range(p):
                # 尝试使用对应层的参数，如果不存在则使用默认值
                gamma_key = f'gamma_{layer}'
                beta_key = f'beta_{layer}'
                
                # 从prev_params中获取参数，如果没有则使用默认值
                gamma_val = prev_params.get(gamma_key, 0.5)
                beta_val = prev_params.get(beta_key, 0.3)
                
                adjusted_params.append(float(gamma_val))
                adjusted_params.append(float(beta_val))
            
            initial_params = np.array(adjusted_params)
            if verbose:
                print(f"使用调整后的热启动参数: {initial_params}")
        else:
            initial_params = None

        training_result = optimizer.optimize_adam_pytorch(
            initial_params=initial_params,
            maxiter=num_steps,
            lr=learning_rate,
            betas=(0.9, 0.999),
            epsilon_fd=0.01,
            patience=early_stop_threshold,
            early_stop_threshold=0.1,
            stop_on_zero_conflict=stop_on_zero_conflict,
            verbose=verbose  # 使用 verbose 参数控制训练日志
        )

        # training_result 是 tuple: (best_params, best_loss, opt_info)
        best_params_array, best_loss, opt_info = training_result

        # 转换为参数字典
        trained_params = {}
        for layer in range(p):
            trained_params[f'gamma_{layer}'] = float(best_params_array[2 * layer])
            trained_params[f'beta_{layer}'] = float(best_params_array[2 * layer + 1])

        training_time = time.time() - start_time

        if verbose or logger:
            log_msg = f"\n[训练完成] k={k}"
            log_msg += f"\n  最优冲突数: {best_loss:.0f}"
            log_msg += f"\n  迭代次数: {opt_info.get('nit', opt_info.get('iterations', 0))}"
            log_msg += f"\n  训练时间: {training_time:.2f}秒"
            log_msg += f"\n  最优参数: {trained_params}"
            if logger:
                logger.info(log_msg)
            if verbose:
                print(log_msg)

        # 如果训练过程中已经找到了无冲突解，直接使用训练中的最优结果
        if best_loss == 0 and opt_info.get('best_coloring') is not None:
            # 训练时已经找到最优解，无需重新采样
            coloring = opt_info.get('best_coloring')
            conflict_count = 0
            query_id = opt_info.get('best_query_id')

            if verbose or logger:
                msg = f"\n✓ 训练过程中已找到无冲突解，直接使用训练最优结果"
                if logger:
                    logger.info(msg)
                if verbose:
                    print(msg)

            if verbose or logger:
                msg = f"[最终结果] k={k}, 冲突数={conflict_count}, 着色方案={coloring}"
                if logger:
                    logger.info(msg)
                if verbose:
                    print(msg)

            training_result_dict = {
                'best_energy': best_loss,
                'best_params': trained_params,
                'energy_history': opt_info.get('loss_history', []),
                'iterations': opt_info.get('nit', opt_info.get('iterations', 0)),
                'converged': True,
                'elapsed_time': training_time,
                'training_mode': 'adam_pytorch'
            }

            return k, coloring, conflict_count, query_id, training_result_dict

        # 训练中没有找到无冲突解，需要重新采样
        best_overall_conflict = float('inf')
        best_overall_coloring = None
        num_sampling_rounds = 10  # 改为只采样10次

        if verbose:
            print(f"\n[使用训练参数执行天衍 QAOA 采样] (采样次数: {num_sampling_rounds})")

        for round_idx in range(num_sampling_rounds):
            try:
                circuit = qaoa_ansatz_tianyan(
                    graph, k, p, trained_params, vertex_colors,
                    nodes_to_recolor, verbose=False, max_qubits=max_qubits
                )
                circuit.measure_all()

                timestamp = datetime.now().strftime("%H%M%S")
                # 使用数据图名称（如果提供）
                if graph_name:
                    base_name = os.path.basename(os.path.splitext(graph_name)[0])
                    # 移除特殊字符，只保留字母、数字、下划线和连字符
                    import re
                    base_name = re.sub(r'[^\w\-]', '_', base_name)
                    exp_name = f'{base_name}_k{k}_p{p}_{timestamp}'
                else:
                    exp_name = f'k{k}_p{p}_{timestamp}'

                # 确保实验名称不为空且有效
                if not exp_name or exp_name.strip() == '':
                    exp_name = f'qaoa_k{k}_p{p}_{timestamp}'
                exp_name = exp_name.strip()

                if verbose or logger:
                    name_msg = f"  实验名称: {exp_name}"
                    if logger:
                        logger.debug(name_msg)
                    if verbose:
                        print(name_msg)

                # 过滤掉 M 门（天衍平台不支持单独的 M 门）
                filtered_qcis = remap_qubits_in_qcis(circuit.qcis, {i: i for i in range(circuit.num_qubits)})

                # 检查过滤后的 QCIS 是否为空
                if filtered_qcis is None or not filtered_qcis or filtered_qcis.strip() == '':
                    if verbose or logger:
                        msg = f"  ✗ 错误：过滤后的 QCIS 为空或None，无法提交实验"
                        if logger:
                            logger.warning(msg)
                        if verbose:
                            print(msg)
                    continue

                submit_kwargs = {
                    'circuit': filtered_qcis,  # 使用过滤后的 QCIS（不含 M 门）
                    'exp_name': exp_name,
                    'num_shots': num_shots
                }
                if lab_id is not None:
                    submit_kwargs['lab_id'] = lab_id

                if verbose or logger:
                    debug_msg = f"  提交实验参数: circuit长度={len(filtered_qcis)}, exp_name={exp_name}, num_shots={num_shots}"
                    if logger:
                        logger.debug(debug_msg)
                    if verbose:
                        print(debug_msg)

                try:
                    query_id = platform.submit_job(**submit_kwargs)
                except Exception as submit_error:
                    error_str = str(submit_error)
                    # 检查是否是机时包限制错误
                    if '机时包' in error_str or 'machine time package' in error_str.lower():
                        if verbose or logger:
                            msg = f"  ✗ 轮次 {round_idx + 1}: 量子比特数超过机时包限制，停止采样"
                            if logger:
                                logger.warning(msg)
                            if verbose:
                                print(msg)
                        # 机时包限制错误，立即停止采样
                        break
                    else:
                        # 其他提交错误，记录并继续
                        if verbose or logger:
                            msg = f"  ✗ 轮次 {round_idx + 1}: 提交实验失败: {error_str}"
                            if logger:
                                logger.warning(msg)
                            if verbose:
                                print(msg)
                        continue

                exp_result = platform.query_experiment(
                    query_id=query_id,
                    max_wait_time=60,
                    sleep_time=1
                )

                if not exp_result or len(exp_result) == 0:
                    continue

                # 检查测量结果，排除全0情况
                result_data = exp_result[0]
                if 'resultStatus' in result_data:
                    measurement_data = result_data['resultStatus']
                    if measurement_data and len(measurement_data) > 1:
                        shots_data = measurement_data[1:]
                        if shots_data:
                            # 转换为字符串格式
                            shots_data_str = []
                            for shot in shots_data:
                                if isinstance(shot, (list, tuple)):
                                    shot_str = ''.join(map(str, shot))
                                else:
                                    shot_str = str(shot)
                                shots_data_str.append(shot_str)

                            # 计算全0态比例
                            total_shots = len(shots_data_str)
                            all_zeros_count = shots_data_str.count('0' * len(shots_data_str[0]))
                            all_zeros_ratio = all_zeros_count / total_shots if total_shots > 0 else 1.0

                            # 统计非全0态采样
                            non_zero_shots = [shot for shot in shots_data_str if shot != '0' * len(shots_data_str[0])]
                            if not non_zero_shots:
                                if verbose or logger:
                                    msg = f"  轮次 {round_idx + 1}: 所有采样均为全0态，无法提取着色，跳过"
                                    if logger:
                                        logger.warning(msg)
                                    if verbose:
                                        print(msg)
                                continue

                            # 如果全0态比例较高，发出警告但继续尝试
                            if all_zeros_ratio > 0.8:
                                if verbose or logger:
                                    msg = f"  轮次 {round_idx + 1}: 全0态比例 {all_zeros_ratio:.1%}，将从{len(non_zero_shots)}个非全0态采样中提取次优解"
                                    if logger:
                                        logger.warning(msg)
                                    if verbose:
                                        print(msg)

                coloring = extract_coloring_tianyan(result_data, graph, k, verbose=False)

                if not coloring:
                    if logger:
                        logger.warning(f"  轮次 {round_idx + 1}: extract_coloring 返回空着色，跳过")
                    continue

                if vertex_colors:
                    for node, color in vertex_colors.items():
                        if node in graph.nodes:
                            coloring[node] = color

                for node in graph.nodes:
                    if graph.degree(node) == 0 and node not in coloring:
                        available_colors = list(coloring.values()) if coloring else [0]
                        coloring[node] = np.random.choice(available_colors)

                current_conflict = count_conflicts(coloring, graph, verbose=False)

                if logger or verbose:
                    msg = f"  轮次 {round_idx + 1}: 冲突数={current_conflict}"
                    if logger:
                        logger.info(msg)
                    if verbose:
                        print(msg)

                if current_conflict < best_overall_conflict:
                    best_overall_conflict = current_conflict
                    best_overall_coloring = coloring.copy()
                    best_query_id = query_id

                    if verbose:
                        print(f"  轮次 {round_idx + 1}: 找到更好方案，冲突数 = {current_conflict}")

                if best_overall_conflict == 0:
                    if verbose:
                        print(f"  轮次 {round_idx + 1}: 找到完美着色方案，提前结束")
                    break

            except Exception as e:
                if verbose:
                    print(f"  轮次 {round_idx + 1} 采样失败: {str(e)}")
                continue

        # 如果重新采样失败，使用训练时的最优着色作为次优解
        if best_overall_conflict == float('inf') and opt_info.get('best_coloring') is not None:
            if logger or verbose:
                msg = f"\n采样阶段未找到有效着色，使用训练时的最优着色作为次优解 (冲突数={best_loss:.0f})"
                if logger:
                    logger.warning(msg)
                if verbose:
                    print(msg)
            coloring = opt_info.get('best_coloring')
            conflict_count = best_loss
            query_id = opt_info.get('best_query_id')
        else:
            coloring = best_overall_coloring or {}
            conflict_count = best_overall_conflict
            query_id = best_query_id if 'best_query_id' in locals() else None

        if logger:
            logger.info(f"[最终结果] k={k}, 冲突数={conflict_count}, 着色方案={coloring}")

        if verbose:
            print(f"\n[最终结果]")
            print(f"  冲突数: {conflict_count}")
            print(f"  着色方案: {coloring}")

        training_result_dict = {
            'best_energy': best_loss,
            'best_params': trained_params,
            'energy_history': opt_info.get('loss_history', []),
            'iterations': opt_info.get('nit', opt_info.get('iterations', 0)),
            'converged': best_loss == 0 or best_loss <= 0.1,
            'elapsed_time': training_time,
            'training_mode': 'adam_pytorch'
        }

        return k, coloring, conflict_count, query_id, training_result_dict

    except Exception as e:
        error_msg = f"天衍真机求解（带训练）出错: {e}"
        if logger:
            logger.error(error_msg)
        if verbose:
            print(error_msg)
            traceback.print_exc()

        training_result_dict = {
            'best_energy': float('inf'),
            'best_params': {},
            'energy_history': [],
            'iterations': 0,
            'converged': False,
            'elapsed_time': 0,
            'training_mode': 'error'
        }

        return k, {}, float('inf'), None, training_result_dict


# ==============================================================================
# 4. 着色辅助函数
# ==============================================================================

def count_conflicts(coloring, graph, verbose=False):
    """
    计算着色方案中的冲突数

    参数:
        coloring: 着色方案字典
        graph: networkx图对象
        verbose: 是否输出详细冲突信息

    返回:
        int: 冲突边数量
    """
    if not isinstance(graph, nx.Graph):
        if verbose:
            print("错误：graph必须是networkx Graph对象")
        return -1

    if not isinstance(coloring, dict):
        if verbose:
            print("错误：coloring必须是字典类型")
        return -1

    if not coloring:
        if verbose:
            print("警告：着色方案为空，冲突数记为0")
        return 0

    valid_nodes = set(coloring.keys())
    graph_nodes = set(graph.nodes)
    missing_nodes = graph_nodes - valid_nodes

    if missing_nodes and verbose:
        print(f"警告：存在未着色的节点 {missing_nodes}")

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

    if verbose and conflicts > 0:
        print(f"冲突边数: {conflicts}")
        print("冲突详情:", conflicting_edges)

    return conflicts


def find_conflict_edges(coloring, graph):
    """高效检测冲突边"""
    conflict_edges = []
    valid_nodes = set(coloring.keys())
    for u, v in graph.edges():
        if u in valid_nodes and v in valid_nodes and coloring[u] == coloring[v]:
            conflict_edges.append((u, v))
    return conflict_edges


def get_cycle_order(cycle_graph):
    """获取环图的节点顺序"""
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

    返回:
        tuple: (coloring, k)
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


def assign_colors_in_order(graph, ordered_nodes, k, vertex_colors=None):
    """
    按指定顺序为图节点分配颜色（贪心策略）

    返回:
        tuple: (coloring, required_k)
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
    验证并调整最小k值

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
    """从最大度节点开始进行贪心着色"""
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


def _greedy_coloring_optimal_k(graph, max_k_limit=50):
    """
    使用贪心算法找到近似最优的着色（自动寻找最小k值）

    对于稀疏图特别有效，能自动使用最少颜色数

    参数:
        graph: NetworkX图对象
        max_k_limit: 最大允许的k值限制

    返回:
        tuple: (着色字典, 使用的颜色数k)
    """
    if not graph or graph.number_of_nodes() == 0:
        return {}, 0

    # 如果是完全图，直接返回n种颜色
    if is_complete_graph(graph):
        n = graph.number_of_nodes()
        coloring = {i: i for i in range(n)}
        return coloring, n

    # 使用DSatur策略：按饱和度（不同颜色的邻居数）排序
    coloring = {}
    uncolored_nodes = set(graph.nodes())

    while uncolored_nodes:
        # 找到饱和度最高的节点（已着色邻居的不同颜色数最多）
        # 如果饱和度相同，选择度数最大的
        best_node = None
        best_saturation = -1
        best_degree = -1

        for node in uncolored_nodes:
            # 计算饱和度
            neighbor_colors = set()
            for neighbor in graph.neighbors(node):
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])
            saturation = len(neighbor_colors)

            # 如果饱和度更高，或者相同但度数更大
            degree = graph.degree(node)
            if (saturation > best_saturation or
                (saturation == best_saturation and degree > best_degree)):
                best_saturation = saturation
                best_degree = degree
                best_node = node

        # 分配最小的可用颜色
        used_colors = set()
        for neighbor in graph.neighbors(best_node):
            if neighbor in coloring:
                used_colors.add(coloring[neighbor])

        # 分配最小可用颜色
        color = 0
        while color in used_colors and color < max_k_limit:
            color += 1

        if color >= max_k_limit:
            # 超过最大限制，分配第一个可用颜色（允许冲突）
            color = 0
            while color in used_colors:
                color += 1

        coloring[best_node] = color
        uncolored_nodes.remove(best_node)

    # 计算实际使用的颜色数
    used_colors_count = len(set(coloring.values())) if coloring else 0

    return coloring, used_colors_count


def _resolve_conflicts_with_greedy(graph, k, max_attempts=100):
    """多次尝试贪心着色直到无冲突或达到最大尝试次数"""
    if k <= 0:
        return None, float('inf')

    if is_complete_graph(graph):
        required_k = len(graph.nodes)
        if k >= required_k:
            coloring = {i: i for i in range(required_k)}
            return coloring, 0
        else:
            return None, float('inf')

    for attempt in range(max_attempts):
        coloring = _greedy_coloring_from_max_degree(graph, k)
        if coloring is None:
            continue

        conflicts = count_conflicts(coloring, graph)
        if conflicts == 0:
            return coloring, 0

    final_coloring = _greedy_coloring_from_max_degree(graph, k) or {}
    final_conflicts = count_conflicts(final_coloring, graph) if final_coloring else float('inf')
    return final_coloring, final_conflicts


def get_subgraph_coloring(subgraph, final_coloring, min_k):
    """
    获取子图着色方案

    返回:
        dict: 子图着色方案字典
    """
    if not subgraph or len(subgraph.nodes) == 0:
        return {}

    if not final_coloring or not isinstance(final_coloring, dict):
        return {node: 0 for node in subgraph.nodes}

    min_k = max(1, min_k)

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
                       canvas_scale=0.8, filename=None, output_dir=None):
    """无着色原始图可视化

    参数:
        graph: NetworkX 图对象
        title: 图标题
        index: 图索引（用于兼容，优先使用 filename）
        layout_seed: 布局随机种子
        node_size: 节点大小
        font_size: 字体大小
        canvas_scale: 画布缩放
        filename: 保存文件名（不含扩展名）
        output_dir: 输出目录路径
    """
    if not graph or len(graph.nodes) == 0:
        print("Warning: Invalid or empty graph, cannot visualize")
        return

    num_nodes = len(graph.nodes)
    fig_width = min(10 + (num_nodes // 10) * 2, 22)
    fig_height = min(8 + (num_nodes // 10) * 1.6, 18)
    plt.figure(figsize=(fig_width, fig_height))

    pos = nx.spring_layout(graph, seed=layout_seed, scale=canvas_scale)
    nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.6, edge_color='gray')

    node_colors = ['#AEC7E8'] * len(graph.nodes)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                           node_size=node_size, edgecolors='#555555',
                           linewidths=1.5)
    nx.draw_networkx_labels(graph, pos, labels={node: str(node) for node in graph.nodes},
                           font_size=font_size, font_family='sans-serif',
                           font_weight='bold')

    plt.title(title, fontsize=14, fontweight='bold', pad=10)
    plt.axis('off')
    plt.tight_layout()

    if filename:
        if output_dir is None:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(BASE_DIR, "graph_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ 原始图已保存: {filepath}")
    elif index is not None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(BASE_DIR, "graph_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"graph_{index}_original.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ 原始图已保存: {filepath}")

    plt.close()


def plot_New_IDs_subgraphs(subgraphs, sub_mappings, title="Divided Subgraphs",
                           filename=None, output_dir=None):
    """可视化子图（使用新编号）

    参数:
        subgraphs: 子图列表
        sub_mappings: 子图节点映射列表
        title: 图标题
        filename: 保存文件名（不含扩展名）
        output_dir: 输出目录路径
    """
    if not subgraphs:
        print("Warning: No subgraphs to visualize")
        return

    num_subgraphs = len([s for s in subgraphs if len(s.nodes) > 0])
    if num_subgraphs == 0:
        return

    cols = min(4, num_subgraphs)
    rows = (num_subgraphs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()

    idx = 0
    for i, (sub, mapping) in enumerate(zip(subgraphs, sub_mappings)):
        if len(sub.nodes) == 0:
            continue

        ax = axes[idx]
        
        # 使用mapping重新映射子图节点为新编号
        if mapping:
            # mapping格式: {原始节点: 新节点}
            # 重新映射子图节点
            sub_renumbered = nx.relabel_nodes(sub, mapping)
        else:
            sub_renumbered = sub
        
        pos = nx.spring_layout(sub_renumbered, seed=i + 42)

        # 绘制矩形边框
        x_values = [pos[node][0] for node in sub_renumbered.nodes]
        y_values = [pos[node][1] for node in sub_renumbered.nodes]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        padding = 0.15
        rect = plt.Rectangle((x_min - padding, y_min - padding),
                          x_max - x_min + 2 * padding,
                          y_max - y_min + 2 * padding,
                          fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7)
        ax.add_patch(rect)

        nx.draw_networkx_edges(sub_renumbered, pos, width=1.5, alpha=0.6, edge_color='gray', ax=ax)
        nx.draw_networkx_nodes(sub_renumbered, pos, node_color='#AEC7E8',
                               node_size=300, edgecolors='#555555',
                               linewidths=1.5, ax=ax)
        # 显示新编号（从0开始，对应量子比特）
        nx.draw_networkx_labels(sub_renumbered, pos,
                               labels={node: str(node) for node in sub_renumbered.nodes},
                               font_size=10, font_family='sans-serif', ax=ax)

        ax.set_title(f"Subgraph {i+1}\nNodes: {len(sub_renumbered.nodes)}, Edges: {len(sub_renumbered.edges)}",
                   fontsize=10, fontweight='bold')
        ax.axis('off')

        idx += 1

    for j in range(idx, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if filename:
        if output_dir is None:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(BASE_DIR, "subgraph_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ 子图（新编号）已保存: {filepath}")

    plt.close()


def plot_Original_IDs_subgraphs(subgraphs, title="Subgraphs with Original Node IDs",
                               filename=None, output_dir=None):
    """可视化子图（使用原始编号）

    参数:
        subgraphs: 子图列表
        title: 图标题
        filename: 保存文件名（不含扩展名）
        output_dir: 输出目录路径
    """
    if not subgraphs:
        print("Warning: No subgraphs to visualize")
        return

    num_subgraphs = len([s for s in subgraphs if len(s.nodes) > 0])
    if num_subgraphs == 0:
        return

    cols = min(4, num_subgraphs)
    rows = (num_subgraphs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()

    idx = 0
    for i, sub in enumerate(subgraphs):
        if len(sub.nodes) == 0:
            continue

        ax = axes[idx]
        pos = nx.spring_layout(sub, seed=i + 42)

        nx.draw_networkx_edges(sub, pos, width=1.5, alpha=0.6, edge_color='gray', ax=ax)
        nx.draw_networkx_nodes(sub, pos, node_color='#AEC7E8',
                               node_size=300, edgecolors='#555555',
                               linewidths=1.5, ax=ax)
        nx.draw_networkx_labels(sub, pos,
                               labels={node: str(node) for node in sub.nodes},
                               font_size=10, font_family='sans-serif', ax=ax)

        ax.set_title(f"Subgraph {i+1}\nNodes: {len(sub.nodes)}, Edges: {len(sub.edges)}",
                   fontsize=10, fontweight='bold')
        ax.axis('off')

        idx += 1

    for j in range(idx, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if filename:
        if output_dir is None:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(BASE_DIR, "subgraph_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ 子图（原始编号）已保存: {filepath}")

    plt.close()


def plot_New_IDs_colored_subgraphs(subgraphs, sub_colorings, sub_mappings, min_k_list=None,
                                   title="Colored Subgraphs (Renumbered)",
                                   filename=None, output_dir=None):
    """可视化带颜色的子图（使用新编号，对应量子比特）"""
    if not subgraphs or not sub_colorings:
        print("Warning: No subgraphs or colorings to visualize")
        return

    num_subgraphs = len(subgraphs)
    cols = min(4, num_subgraphs)
    rows = (num_subgraphs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()

    for i, (sub, coloring) in enumerate(zip(subgraphs, sub_colorings)):
        if len(sub.nodes) == 0:
            continue

        ax = axes[i]
        
        # 使用mapping重新映射子图节点为新编号
        mapping = sub_mappings[i] if sub_mappings and i < len(sub_mappings) else None
        if mapping:
            # mapping格式: {原始节点: 新节点}
            # 重新映射子图节点
            sub_renumbered = nx.relabel_nodes(sub, mapping)
            # 重新映射着色字典的键到新节点编号
            coloring_renumbered = {mapping[old]: color for old, color in coloring.items() if old in mapping}
        else:
            sub_renumbered = sub
            coloring_renumbered = coloring
        
        pos = nx.spring_layout(sub_renumbered, seed=i + 42)

        # 绘制矩形边框
        x_values = [pos[node][0] for node in sub_renumbered.nodes]
        y_values = [pos[node][1] for node in sub_renumbered.nodes]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        padding = 0.15
        rect = plt.Rectangle((x_min - padding, y_min - padding),
                          x_max - x_min + 2 * padding,
                          y_max - y_min + 2 * padding,
                          fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7)
        ax.add_patch(rect)

        node_colors = [coloring_renumbered.get(node, 0) for node in sub_renumbered.nodes]
        max_k = min_k_list[i] if min_k_list and i < len(min_k_list) else max(node_colors) + 1
        cmap = plt.cm.get_cmap('tab10', max(max_k, 3))

        # 检测冲突边
        conflict_edges = [(u, v) for u, v in sub_renumbered.edges() if coloring_renumbered.get(u, 0) == coloring_renumbered.get(v, 0)]
        normal_edges = [(u, v) for u, v in sub_renumbered.edges() if coloring_renumbered.get(u, 0) != coloring_renumbered.get(v, 0)]

        # 绘制普通边（灰色）
        if normal_edges:
            nx.draw_networkx_edges(sub_renumbered, pos, edgelist=normal_edges, width=1.5, alpha=0.6, edge_color='gray', ax=ax)

        # 绘制冲突边（红色加粗）
        if conflict_edges:
            nx.draw_networkx_edges(sub_renumbered, pos, edgelist=conflict_edges, width=3.0, alpha=1.0, edge_color='red', ax=ax)

        nx.draw_networkx_nodes(sub_renumbered, pos, node_color=node_colors, cmap=cmap,
                               node_size=300, edgecolors='#555555',
                               linewidths=1.5, ax=ax)
        # 显示新编号（从0开始，对应量子比特）
        nx.draw_networkx_labels(sub_renumbered, pos,
                               labels={node: str(node) for node in sub_renumbered.nodes},
                               font_size=10, font_family='sans-serif', ax=ax)

        conflicts = len(conflict_edges)
        ax.set_title(f"Subgraph {i+1}\nNodes: {len(sub_renumbered.nodes)}, Edges: {len(sub_renumbered.edges)}\nk={max_k}, Conflicts: {conflicts}",
                   fontsize=10, fontweight='bold')
        ax.axis('off')

    for j in range(num_subgraphs, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if filename:
        if output_dir is None:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(BASE_DIR, "subgraph_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ 着色子图（新编号）已保存: {filepath}")

    plt.close()


def plot_Original_IDs_colored_subgraphs(
    subgraphs, sub_colorings, title="Colored Subgraphs (Original IDs)",
    min_k_list=None, filename=None, output_dir=None
):
    """可视化带颜色的子图（使用原始编号）"""
    if not subgraphs or not sub_colorings:
        print("Warning: No subgraphs or colorings to visualize")
        return

    num_subgraphs = len(subgraphs)
    cols = min(4, num_subgraphs)
    rows = (num_subgraphs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()

    for i, (sub, coloring) in enumerate(zip(subgraphs, sub_colorings)):
        if len(sub.nodes) == 0:
            continue

        ax = axes[i]
        pos = nx.spring_layout(sub, seed=i + 42)

        node_colors = [coloring.get(node, 0) for node in sub.nodes]
        max_k = min_k_list[i] if min_k_list and i < len(min_k_list) else max(node_colors) + 1
        cmap = plt.cm.get_cmap('tab10', max(max_k, 3))

        # 检测冲突边
        conflict_edges = [(u, v) for u, v in sub.edges() if coloring.get(u, 0) == coloring.get(v, 0)]
        normal_edges = [(u, v) for u, v in sub.edges() if coloring.get(u, 0) != coloring.get(v, 0)]

        # 绘制普通边（灰色）
        if normal_edges:
            nx.draw_networkx_edges(sub, pos, edgelist=normal_edges, width=1.5, alpha=0.6, edge_color='gray', ax=ax)

        # 绘制冲突边（红色加粗）
        if conflict_edges:
            nx.draw_networkx_edges(sub, pos, edgelist=conflict_edges, width=3.0, alpha=1.0, edge_color='red', ax=ax)

        nx.draw_networkx_nodes(sub, pos, node_color=node_colors, cmap=cmap,
                               node_size=300, edgecolors='#555555',
                               linewidths=1.5, ax=ax)
        nx.draw_networkx_labels(sub, pos,
                               labels={node: str(node) for node in sub.nodes},
                               font_size=10, font_family='sans-serif', ax=ax)

        conflicts = len(conflict_edges)
        ax.set_title(f"Subgraph {i+1}\nNodes: {len(sub.nodes)}, Edges: {len(sub.edges)}\nk={max_k}, Conflicts: {conflicts}",
                   fontsize=10, fontweight='bold')
        ax.axis('off')

    for j in range(num_subgraphs, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if filename:
        if output_dir is None:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(BASE_DIR, "subgraph_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ 着色子图（原始编号）已保存: {filepath}")

    plt.close()


def visualize_graph(graph, coloring=None, title="Graph Visualization", index=None,
                   min_k=None, filename=None, processing_time=None,
                   layout_seed=42, node_size=300, font_size=12,
                   canvas_scale=0.8, output_dir=None):
    """可视化图（可选着色）

    参数:
        graph: NetworkX 图对象
        coloring: 可选着色方案字典 {node: color}
        title: 图标题
        index: 图索引（用于文件命名）
        min_k: 最小颜色数
        filename: 保存文件名（不含扩展名）
        processing_time: 处理时间（秒）
        layout_seed: 布局随机种子
        node_size: 节点大小
        font_size: 字体大小
        canvas_scale: 画布缩放
        output_dir: 输出目录路径
    """
    if not graph or len(graph.nodes) == 0:
        print("Warning: Invalid or empty graph, cannot visualize")
        return

    num_nodes = len(graph.nodes)
    fig_width = min(10 + (num_nodes // 10) * 2, 22)
    fig_height = min(8 + (num_nodes // 10) * 1.6, 18)
    plt.figure(figsize=(fig_width, fig_height))

    pos = nx.spring_layout(graph, seed=layout_seed, scale=canvas_scale)

    if coloring is None:
        node_colors = ['#AEC7E8'] * num_nodes
        # 没有着色时，不检测冲突
        conflict_edges = []
    else:
        node_colors = [coloring.get(node, 0) for node in graph.nodes]
        max_k = min_k if min_k else max(node_colors) + 1
        node_colors = plt.cm.get_cmap('tab10', max(max_k, 3))(node_colors)

        # 检测冲突边：两个端点颜色相同的边
        conflict_edges = [(u, v) for u, v in graph.edges() if coloring.get(u, 0) == coloring.get(v, 0)]

        # 分别绘制普通边和冲突边
        normal_edges = [(u, v) for u, v in graph.edges() if coloring.get(u, 0) != coloring.get(v, 0)]

        # 绘制普通边（灰色）
        if normal_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=normal_edges, width=1.5, alpha=0.6, edge_color='gray')

        # 绘制冲突边（红色加粗）
        if conflict_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=conflict_edges, width=3.0, alpha=1.0, edge_color='red')

    if coloring is None:
        # 没有着色时，绘制所有边为灰色
        nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.6, edge_color='gray')

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                           node_size=node_size, edgecolors='#555555',
                           linewidths=1.5)
    nx.draw_networkx_labels(graph, pos, labels={node: str(node) for node in graph.nodes},
                           font_size=font_size, font_family='sans-serif',
                           font_weight='bold')

    if coloring:
        conflicts = len(conflict_edges) if conflict_edges else count_conflicts(coloring, graph)
        subtitle = f"Colors Used: {len(set(coloring.values()))}, Conflicts: {conflicts}"
        if processing_time:
            subtitle += f", Time: {processing_time:.2f}s"
    else:
        subtitle = "No Coloring"

    plt.title(f"{title}\n{subtitle}", fontsize=14, fontweight='bold', pad=10)
    plt.axis('off')
    plt.tight_layout()

    if filename:
        if output_dir is None:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(BASE_DIR, "graph_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ 图已保存: {filepath}")

    plt.close()


def smart_divide_graph_with_qubit_constraint(
    graph,
    max_qubits=36,
    max_k_per_subgraph=20,
    Q=20
):
    """
    智能子图划分: 根据量子比特约束自动调整子图大小
    适配天衍平台，避免退化为贪心着色

    策略:
    1. 根据max_k计算每节点需要的比特数: bits_per_node = ceil(log2(max_k))
    2. 计算最大允许节点数: max_nodes = max_qubits // bits_per_node
    3. 初始子图数: ceil(total_nodes / max_nodes)
    4. 递归二分超限子图

    参数:
        graph: 待划分的图
        max_qubits: 最大量子比特数 (天衍平台默认36)
        max_k_per_subgraph: 每个子图最大颜色数 (默认20)
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

        # 安全限制：最大递归深度为20，确保不会无限递归
        if depth >= 20:
            logging.warning(f"递归深度超过限制（depth={depth}），强制输出子图（{n}节点，{n * bits_per_node}比特）")
            final_subgraphs.append(g)
            final_mappings.append(mapping)
            info['subgraph_stats'].append({
                'nodes': n,
                'edges': g.number_of_edges(),
                'qubits_required': n * bits_per_node,
                'bits_per_node': bits_per_node,
                'feasible': False,  # 超出深度限制，标记为不可行
                'warning': 'max_depth_exceeded'
            })
            return

        # 计算所需量子比特数
        required_qubits = n * bits_per_node

        # 如果子图满足约束，或者非常小（<=3节点），直接添加
        if required_qubits <= max_qubits or n <= 3:
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

        # 超出量子比特限制，需要二分
        logging.warning(f"子图{n}节点需要{required_qubits}比特 > {max_qubits}限制，进行二分(深度{depth})")

        # 尝试使用METIS二分
        try:
            _, part2 = metis.part_graph(g, 2, objtype="cut", ufactor=Q * 10)
        except Exception as e:
            logging.warning(f"METIS二分失败（深度{depth}），使用简单二分: {e}")
            # METIS失败，简单按节点数量二分
            nodes_list = sorted(list(g.nodes))
            mid = len(nodes_list) // 2
            part2 = [0] * mid + [1] * (len(nodes_list) - mid)

        # 提取两个子图的节点
        nodes0 = [n for n, p in zip(g.nodes, part2) if p == 0]
        nodes1 = [n for n, p in zip(g.nodes, part2) if p == 1]

        # 检查是否有空子图
        if len(nodes0) == 0:
            logging.warning(f"二分产生空子图，跳过nodes0（深度{depth}）")
            nodes0 = nodes_list[:1]  # 至少保留一个节点

        if len(nodes1) == 0:
            logging.warning(f"二分产生空子图，跳过nodes1（深度{depth}）")
            nodes1 = nodes_list[-1:]  # 至少保留一个节点

        # 递归处理两个子图
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


def handle_exception(func_name, index, e):
    """统一的异常处理函数"""
    print(f"Error in {func_name} for graph {index}: {str(e)}")
    traceback.print_exc()




