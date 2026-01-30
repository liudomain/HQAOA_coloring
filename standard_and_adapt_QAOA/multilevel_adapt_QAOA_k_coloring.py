"""
自适应QAOA多层次图着色模块

本模块实现基于自适应QAOA(Adaptive QAOA)的多层次图着色算法，主要包含：
1. solve_k_coloring: 使用自适应QAOA求解k着色问题
2. sequential_process_subgraphs: 顺序处理子图着色
3. iterative_optimization: 迭代优化全局着色方案

算法特点：
- 孤立节点批量处理（快速路径）
- 环图专用着色算法（QAOA不参与）
- 普通图采用自适应QAOA+贪心混合策略
- 同构子图缓存复用
- 固定k值多次重试优化
"""

# ==============================================================================
# 导入模块
# ==============================================================================

import csv  # 补充缺失的导入
import copy
import time
import traceback
import json
import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import mindspore
from mindspore import nn, Tensor
from mindquantum import Circuit, ParameterResolver, MQAnsatzOnlyLayer, H, UN, RX, GlobalPhase, Measure
from mindquantum.simulator import Simulator
import mindspore as ms
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mindquantum.core.operators import TimeEvolution, Hamiltonian, QubitOperator
import matplotlib
# 使用非交互式后端，图片显示后不阻塞程序继续执行
matplotlib.use('Agg')
from matplotlib import cm
from math import log2, ceil

# 从共享模块导入通用函数
from multilevel_common import (
    BASE_DIR, LOGS_DIR,
    divide_graph, count_conflicts, extract_coloring,
    is_complete_graph, is_odd_cycle, is_cycle_graph,
    get_graph_signature, setup_logger,
    get_cycle_order, cycle_graph_coloring, assign_colors_in_order, validate_min_k,
    build_hamiltonian, adapt_qaoa_ansatz,
    mixer_pool_single, mixer_pool_multi, build_mixer_pool,
    qaoa_cost, qaoa_mixer, derivative, find_conflict_edges,
    plot_original_graph, plot_New_IDs_subgraphs, plot_Original_IDs_subgraphs,
    _greedy_coloring_from_max_degree, _resolve_conflicts_with_greedy,
    plot_New_IDs_colored_subgraphs, plot_Original_IDs_colored_subgraphs,
    get_subgraph_coloring, visualize_graph, handle_exception
)

# from multilevel_common_improved import (smart_divide_graph_with_qubit_constraint)
# ==============================================================================
# 1. 核心QAOA求解函数
# ==============================================================================

def solve_k_coloring(graph, k, p=1, num_steps=1000, vertex_colors=None,
                     nodes_to_recolor=None, penalty= 1000, Q=20,
                     learning_rate=0.1, early_stop_threshold=2, prev_params=None):
    """
    使用自适应QAOA算法求解图的k着色问题

    算法流程：
    1. 特殊图快速处理：空图、单节点图直接返回
    2. 构建自适应QAOA量子线路和哈密顿量
    3. 使用学习率衰减和热启动进行参数优化
    4. 多次采样寻找最优着色方案

    参数:
        graph: 待着色的图（networkx图对象）
        k: 尝试使用的颜色数
        p: QAOA算法的层数（电路深度）
        num_steps: 最大训练迭代步数
        vertex_colors: 顶点颜色的节点字典 {节点: 颜色}
        nodes_to_recolor: 需要重新着色的节点列表
        penalty: 冲突惩罚系数
        Q: 图划分相关参数
        learning_rate: 优化器初始学习率
        early_stop_threshold: 连续相同训练值的提前退出阈值
        prev_params: 热启动参数（上一轮k值的最优参数）

    返回:
        tuple: (best_k, conv_param, best_coloring, conflict_history, best_params)
            - best_k: 最佳颜色数
            - conv_param: 训练收敛参数值（最终训练损失值，保留4位小数）
            - best_coloring: 最佳着色方案字典 {节点: 颜色}
            - conflict_history: 训练历史记录列表（包含每步训练值）
            - best_params: 最优QAOA参数
    """

    # ======================================================================
    # 1.1 特殊图快速处理
    # ======================================================================
    # 空图：返回默认值
    if len(graph.nodes) == 0:
        return 0, 0.0, {}, [], None

    # 单节点图：直接分配颜色
    if len(graph.nodes) == 1:
        node = list(graph.nodes)[0]
        color = vertex_colors[node] if (vertex_colors and node in vertex_colors) else 0
        return 1, 0.0, {node: color}, [], None

    # ======================================================================
    # 1.2 初始化变量
    # ======================================================================
    conflict_history = []
    best_k = k
    best_coloring = None
    best_conflict = float('inf')
    consecutive_same = 0  # 连续相同训练值计数
    prev_train_value = None
    final_train_value = 0.0
    best_params = None

    try:
        # -----------------------------------------------------------------
        # 构建QAOA量子线路和哈密顿量
        # -----------------------------------------------------------------
        circ = adapt_qaoa_ansatz(graph, k, p, vertex_colors, nodes_to_recolor, penalty)
        sim = Simulator('mqvector', circ.n_qubits)
        ham = build_hamiltonian(
            graph, k, vertex_colors, nodes_to_recolor, penalty
        )
        
        # 调试：打印哈密顿量信息
        n_qubits = math.ceil(math.log2(k))
        num_edges = graph.number_of_edges()
        print(f"[DEBUG] k={k}, n_qubits={n_qubits}, num_edges={num_edges}, penalty={penalty}")
        print(f"[DEBUG] 理论最小值=0, 理论最大值={num_edges * n_qubits * penalty}")
        
        grad_ops = sim.get_expectation_with_grad(ham, circ)

        # -----------------------------------------------------------------
        # 学习率衰减优化器
        # -----------------------------------------------------------------
        lr_scheduler = nn.exponential_decay_lr(
            learning_rate=learning_rate,
            decay_rate=0.9,
            total_step=num_steps,
            step_per_epoch=20,
            decay_epoch=1
        )
        net = MQAnsatzOnlyLayer(grad_ops)
        opti = nn.Adam(net.trainable_params(), learning_rate=Tensor(lr_scheduler))
        train_net = nn.TrainOneStepCell(net, opti, sens=1.0)  # 梯度裁剪

        # 热启动：使用上一轮k值的最优参数
        if prev_params is not None and len(prev_params) == len(circ.params_name):
            print(f"使用热启动参数初始化k={k}的训练")
            net.weight.set_data(ms.Tensor(prev_params, dtype=ms.float32))

        current_best_conflict = float('inf')
        current_best_params = None

        # -----------------------------------------------------------------
        # 训练迭代（带提前退出）
        # -----------------------------------------------------------------
        for step in range(num_steps):
            # 执行训练步骤
            raw_train_value = train_net()

            # 处理训练值（MindQuantum 返回负期望值，取绝对值）
            if isinstance(raw_train_value, ms.Tensor):
                train_value = abs(raw_train_value.asnumpy().item())
            else:
                print(f"步骤 {step} 警告: 训练值类型异常({type(raw_train_value)})")
                continue

            # 记录训练值和历史
            final_train_value = train_value
            conflict_history.append((k, step, train_value))

            # 检查提前退出条件：连续early_stop_threshold步训练值在误差范围内相同
            # 使用0.1作为容差，当连续多次训练值变化小于0.1时认为收敛
            if prev_train_value is not None and abs(train_value - prev_train_value) <= 0.1:
                consecutive_same += 1
                if consecutive_same >= early_stop_threshold:
                    print(f"k={k} 提前退出: 连续{early_stop_threshold}步训练值变化≤0.1({train_value:.1f})")
                    break
            else:
                consecutive_same = 0
                prev_train_value = train_value

            # 定期输出训练进度
            if step % 20 == 0:
                print(f"k={k}, 步骤 {step}/{num_steps}: 训练值 {train_value:.1f}")

            # 更新当前最佳参数
            if train_value < current_best_conflict:
                current_best_conflict = train_value
                current_best_params = net.weight.asnumpy().copy()

        # -----------------------------------------------------------------
        # 采样寻找最优着色
        # -----------------------------------------------------------------
        if current_best_params is not None:
            best_params = net.weight.asnumpy().copy()
            pr = dict(zip(circ.params_name, current_best_params))

            # 多次采样寻找最优着色（100次）
            for _ in range(100):
                try:
                    # 构建测量电路
                    temp_circ = circ.copy()
                    for qubit in range(circ.n_qubits):
                        temp_circ.measure(qubit)

                    # 执行采样
                    result = sim.sampling(temp_circ, pr, shots=1000)
                    coloring = extract_coloring(result, graph, k)

                    if coloring is None:
                        continue

                    # 处理顶点颜色节点（确保不被覆盖）
                    if vertex_colors:
                        for node, color in vertex_colors.items():
                            if node in graph.nodes:
                                coloring[node] = color

                    # 处理孤立节点（度为0的节点）
                    for node in graph.nodes:
                        if graph.degree(node) == 0 and node not in coloring:
                            available_colors = list(coloring.values()) if coloring else [0]
                            coloring[node] = np.random.choice(available_colors)

                    # 计算当前着色的冲突数
                    current_conflict = count_conflicts(coloring, graph)

                    # 更新全局最优解
                    if current_conflict < best_conflict or (
                            current_conflict == best_conflict and k < best_k):
                        best_k = k
                        best_coloring = coloring.copy()
                        best_conflict = current_conflict

                    # 找到完美着色（无冲突）可提前结束采样
                    if best_conflict == 0:
                        print(f"k={k} 找到完美着色方案，冲突数=0")
                        break

                except Exception as e:
                    print(f"采样过程出错: {str(e)}")
                    continue

        # 将最终训练值添加到历史记录
        conflict_history.append(('final', k, final_train_value))

    except Exception as e:
        print(f"k={k} 计算失败: {str(e)}")
        return None, final_train_value, None, conflict_history, best_params

    return best_k, round(final_train_value, 4), best_coloring, conflict_history, best_params


# ==============================================================================
# 2. 顺序子图处理函数
# ==============================================================================

def sequential_process_subgraphs(
    subgraphs,
    sub_mappings,
    dataset_name,
    graph_id,
    max_k=20,
    p=1,
    num_steps=1000,
    vertex_colors=None,
    nodes_to_recolor=None,
    penalty= 1000,
    Q=20,
    learning_rate=0.01
):
    """
    顺序处理子图着色的核心函数

    处理策略：
    1. 孤立节点批量处理（快速路径）
    2. 环图专用着色（QAOA不参与）
    3. 普通图：自适应QAOA + 贪心混合策略
    4. 同构子图缓存复用
    5. 固定k值多次重试优化

    参数:
        subgraphs: 子图列表
        sub_mappings: 子图节点映射列表
        dataset_name: 数据集名称
        graph_id: 图ID
        max_k: 最大尝试的颜色数
        p: QAOA层数
        num_steps: 最大训练步数
        vertex_colors: 顶点颜色字典
        nodes_to_recolor: 需要重新着色的节点列表
        penalty: 冲突惩罚系数
        Q: 图划分参数
        learning_rate: 学习率

    返回:
        list: 子图结果列表，每个元素为 (min_k, coloring, conflict_count, status, retry_info)
            - min_k: 最小颜色数
            - coloring: 着色方案字典
            - conflict_count: 冲突数
            - status: 状态标识
            - retry_info: 重试信息字典
    """

    # ======================================================================
    # 2.1 初始化日志
    # ======================================================================
    logger = setup_logger(dataset_name, graph_id)

    # ======================================================================
    # 2.2 初始化核心变量
    # ======================================================================
    results = [None] * len(subgraphs)
    signature_cache = {}  # 同构子图缓存 {signature: (k, coloring, max_color)}
    processed_subgraphs = []

    # ======================================================================
    # 2.3 批量处理孤立节点子图（无边）
    # ======================================================================
    isolated_subgraphs = [
        (i, sub, mapping)
        for i, (sub, mapping) in enumerate(zip(subgraphs, sub_mappings))
        if len(sub.nodes) > 0 and len(sub.edges) == 0
    ]

    if isolated_subgraphs:
        logger.info(f"\n===== 开始集中处理 {len(isolated_subgraphs)} 个孤立节点子图 =====")
        batch_start_time = time.time()

        for i, sub, mapping in isolated_subgraphs:
            sub_start_time = time.time()
            try:
                new_sub = nx.relabel_nodes(sub, mapping)
                global_coloring = {node: 0 for node in new_sub.nodes}
                original_coloring = {
                    old: global_coloring[new]
                    for old, new in mapping.items()
                    if new in global_coloring
                }
                processing_time = time.time() - sub_start_time
                results[i] = (
                    1,
                    original_coloring,
                    0,
                    'success (isolated nodes, batch processed)',
                    {'batch_processed': True, 'total_attempts': 1, 'success_attempt': 1, 'conflict_history': [0]}
                )
                processed_subgraphs.append(i)
                logger.info(f"孤立子图 {i + 1} 完成：节点数={len(sub.nodes)}，k=1，冲突数=0")

            except Exception as e:
                processing_time = time.time() - sub_start_time
                logger.error(f"孤立子图 {i + 1} 处理失败: {str(e)}")
                results[i] = (1, {}, float('inf'), 'failed (mapping error)', {'error': str(e), 'total_attempts': 1})

        batch_time = time.time() - batch_start_time
        logger.info(f"===== 孤立节点子图处理完成，总耗时: {batch_time:.1f}秒 =====")

    # ======================================================================
    # 2.4 内部辅助函数
    # ======================================================================
    def check_adjacent_subgraphs(current_idx):
        """检查当前子图是否有相邻的已处理子图"""
        if current_idx < 0 or current_idx >= len(subgraphs):
            return None
        if current_idx > 0 and (current_idx - 1) in processed_subgraphs:
            return current_idx - 1
        if current_idx < len(subgraphs) - 1 and (current_idx + 1) in processed_subgraphs:
            return current_idx + 1
        return None

    def complement_coloring(coloring, max_color):
        """生成互补着色方案，用于相邻子图"""
        if not coloring or max_color <= 0:
            return coloring.copy()
        offset = (max_color + 1) // 2
        return {node: (color + offset) % (max_color + 1) for node, color in coloring.items()}

    # ======================================================================
    # 2.5 主循环：处理非孤立子图
    # ======================================================================
    for i, (sub, mapping) in enumerate(zip(subgraphs, sub_mappings)):
        if results[i] is not None:
            continue

        sub_start_time = time.time()
        sub_result = {
            'min_k': 1,
            'coloring': {},
            'conflict_count': float('inf'),
            'status': 'failed',
            'retry_info': {
                'total_attempts': 0,
                'conflict_history': [],
                'success_attempt': None
            }
        }

        # -----------------------------------------------------------------
        # 空子图处理
        # -----------------------------------------------------------------
        if len(sub.nodes) == 0:
            processing_time = time.time() - sub_start_time
            results[i] = (1, {}, 0, 'skipped (empty subgraph)', {})
            continue

        # -----------------------------------------------------------------
        # 节点映射
        # -----------------------------------------------------------------
        try:
            new_sub = nx.relabel_nodes(sub, mapping)
        except Exception as e:
            processing_time = time.time() - sub_start_time
            logger.error(f"子图 {i + 1} 节点映射失败: {str(e)}")
            results[i] = (1, {}, float('inf'), 'failed (mapping error)', sub_result['retry_info'])
            continue

        if len(new_sub.nodes) == 0:
            processing_time = time.time() - sub_start_time
            results[i] = (1, {}, 0, 'skipped (empty after mapping)', {})
            continue

        # -----------------------------------------------------------------
        # 环图专用着色（QAOA不参与）
        # -----------------------------------------------------------------
        if is_cycle_graph(new_sub):
            n = new_sub.number_of_nodes()
            cycle_type = "Even_ring" if n % 2 == 0 else "Odd_ring"
            logger.info(f"\n___________处理子图 {i + 1}（{cycle_type}，节点数：{n}）____________")
            logger.info(f"检测到{cycle_type}，使用环图专用着色（QAOA不参与）")

            try:
                # 使用环图专用着色算法
                cycle_coloring, cycle_k = cycle_graph_coloring(new_sub)
                conflict_count = count_conflicts(cycle_coloring, new_sub)
                if conflict_count != 0:
                    raise RuntimeError(f"环图着色异常！预期0冲突，实际{conflict_count}冲突")

                # 映射回原始节点ID
                reverse_mapping = {new: old for old, new in mapping.items()}
                original_coloring = {}
                for new_node, color in cycle_coloring.items():
                    if new_node in reverse_mapping:
                        original_coloring[reverse_mapping[new_node]] = color

                # 更新子图结果
                sub_result.update({
                    'min_k': cycle_k,
                    'coloring': original_coloring,
                    'conflict_count': conflict_count,
                    'status': 'success (cycle graph)',
                    'retry_info': {'total_attempts': 1, 'success_attempt': 1, 'conflict_history': [0]}
                })
                sub_signature = get_graph_signature(new_sub)
                signature_cache[sub_signature] = (cycle_k, cycle_coloring, max(cycle_coloring.values()))

                results[i] = (
                    sub_result['min_k'],
                    sub_result['coloring'],
                    sub_result['conflict_count'],
                    sub_result['status'],
                    sub_result['retry_info']
                )
                processed_subgraphs.append(i)
                processing_time = time.time() - sub_start_time
                logger.info(f"子图 {i + 1} 完成：k={cycle_k}，冲突数=0，状态=success (cycle graph)")

                continue

            except Exception as e:
                logger.warning(f"环图着色失败: {str(e)}，切换贪心着色兜底")
                # 贪心着色兜底
                reverse_mapping = {new: old for old, new in mapping.items()}
                sorted_nodes = sorted(new_sub.nodes, key=lambda x: new_sub.degree(x), reverse=True)
                greedy_coloring, required_k = assign_colors_in_order(
                    new_sub, sorted_nodes, k=3 if n % 2 else 2
                )
                conflict_count = count_conflicts(greedy_coloring, new_sub)
                original_coloring = {
                    reverse_mapping[new]: greedy_coloring[new]
                    for new in greedy_coloring
                    if new in reverse_mapping
                }

                sub_result.update({
                    'min_k': required_k,
                    'coloring': original_coloring,
                    'conflict_count': conflict_count,
                    'status': 'success (cycle fallback greedy)',
                    'retry_info': {
                        'error': str(e),
                        'total_attempts': 1,
                        'conflict_history': [conflict_count],
                        'success_attempt': 1 if conflict_count == 0 else None
                    }
                })
                processing_time = time.time() - sub_start_time
                results[i] = (
                    sub_result['min_k'],
                    sub_result['coloring'],
                    sub_result['conflict_count'],
                    sub_result['status'],
                    sub_result['retry_info']
                )
                processed_subgraphs.append(i)
                logger.info(f"子图 {i + 1} 兜底完成：k={required_k}，冲突数={conflict_count}")

                continue

        # -----------------------------------------------------------------
        # 普通图处理（QAOA + 贪心）
        # -----------------------------------------------------------------
        logger.info(
            f"\n___________处理子图 {i + 1}（普通图，节点数：{len(new_sub.nodes)}，边数：{len(new_sub.edges)}）____________")

        # 检查缓存复用
        sub_signature = get_graph_signature(new_sub)
        logger.info(f"子图 {i + 1} 签名: {sub_signature[:8]}...")

        if sub_signature in signature_cache:
            # 同构子图缓存复用
            cached_k, cached_coloring, cached_max_color = signature_cache[sub_signature]
            logger.info(f"发现同构子图，复用色数={cached_k}的结果")

            adjacent_idx = check_adjacent_subgraphs(i)
            if adjacent_idx is not None:
                logger.info(f"与已处理子图 {adjacent_idx + 1} 相邻，应用互补着色")
                reused_coloring = complement_coloring(cached_coloring, cached_max_color)
            else:
                reused_coloring = cached_coloring.copy()

            original_coloring = {
                old: reused_coloring[new]
                for old, new in mapping.items()
                if new in reused_coloring
            }
            conflict_count = count_conflicts(original_coloring, sub)

            sub_result.update({
                'min_k': cached_k,
                'coloring': original_coloring,
                'conflict_count': conflict_count,
                'status': 'success (cached)',
                'retry_info': {'total_attempts': 1, 'conflict_history': [conflict_count],
                               'success_attempt': 1 if conflict_count == 0 else None}
            })
            processing_time = time.time() - sub_start_time
            results[i] = (
                sub_result['min_k'],
                sub_result['coloring'],
                sub_result['conflict_count'],
                sub_result['status'],
                sub_result['retry_info']
            )
            processed_subgraphs.append(i)
            logger.info(f"子图 {i + 1} 缓存复用完成：k={cached_k}，冲突数={conflict_count}")

            continue

        # -----------------------------------------------------------------
        # 计算理论最小色数（Brooks定理）
        # -----------------------------------------------------------------
        # Brooks定理：如果图G不是完全图也不是奇环，那么色数不超过最大度（Δ），即χ(G) ≤ Δ
        n = new_sub.number_of_nodes()
        max_degree = max(new_sub.degree(node) for node in new_sub.nodes) if n > 1 else 0

        # ======================================================================
        # 判断子图复杂度，决定是否必须使用QAOA
        # ======================================================================
        # 复杂图定义（必须使用QAOA）：
        # 1. 边数 >= 5
        # 2. 节点数 >= 3 且 平均度 >= 1.5
        # 3. 最大度 >= 3
        # 这样可以避免对稀疏子图直接退化为贪心算法
        is_complex_subgraph = (
            len(new_sub.edges) >= 5 or
            (len(new_sub.nodes) >= 3 and sum(dict(new_sub.degree()).values()) / len(new_sub.nodes) >= 1.5) or
            max_degree >= 3
        )

        if is_complex_subgraph:
            logger.info(f"子图判定为复杂图（边数={len(new_sub.edges)}，节点数={len(new_sub.nodes)}，最大度={max_degree}），强制使用QAOA")
        else:
            logger.info(f"子图为简单图（边数={len(new_sub.edges)}，节点数={len(new_sub.nodes)}，最大度={max_degree}），允许快速路径")
        is_complete = is_complete_graph(new_sub)

        if is_complete:
            theoretical_min_k = n
            logger.info(f"子图是完全图，理论最小色数={theoretical_min_k}")
        else:
            theoretical_min_k = max_degree
            logger.info(f"子图是普通图（最大度Δ={max_degree}），理论最小色数≤{theoretical_min_k}")

        max_test_k = min(theoretical_min_k, max_k)
        max_test_k = max(max_test_k, 2)

        # 复杂图禁用单 k 值快速路径，强制使用QAOA多k值尝试
        if max_test_k == 2 and theoretical_min_k == 2 and not is_complex_subgraph:
            logger.info("仅尝试 k=2（单 k 值快速路径）")
            k_candidates = [2]
        else:
            if is_complex_subgraph:
                logger.info(f"复杂图强制使用QAOA，k值范围: 2 ~ {max_test_k}")
            else:
                logger.info(f"QAOA尝试k值范围: 2 ~ {max_test_k}")
            k_candidates = range(2, max_test_k + 1)

        # -----------------------------------------------------------------
        # 遍历k值，用自适应QAOA求解
        # -----------------------------------------------------------------
        k_results = []
        found_zero_conflict = False
        best_zero_k = None

        for k in k_candidates:
            if found_zero_conflict:
                logger.info(f"已找到0冲突K值（{best_zero_k}），终止后续K值训练")
                break

            logger.info(f"\n尝试k={k}着色...")
            try:
                qaoa_result = solve_k_coloring(
                    graph=new_sub,
                    k=k,
                    p=p,
                    num_steps=num_steps,
                    vertex_colors=vertex_colors,
                    nodes_to_recolor=nodes_to_recolor,
                    penalty=penalty,
                    Q=Q,
                    learning_rate=learning_rate,
                    early_stop_threshold=2
                )

                if not (qaoa_result and len(qaoa_result) == 5):
                    logger.warning(f"k={k} QAOA结果无效（长度≠5）")
                    continue

                _, conv_param, qaoa_coloring, _, _ = qaoa_result

                if not (isinstance(qaoa_coloring, dict) and qaoa_coloring):
                    logger.warning(f"k={k} QAOA未生成有效着色方案")
                    continue

                current_conflict = count_conflicts(qaoa_coloring, new_sub)
                k_results.append((k, conv_param, current_conflict, qaoa_coloring))
                logger.info(f"k={k} 完成：收敛参数={conv_param:.4f}，冲突数={current_conflict}")

                if current_conflict == 0:
                    found_zero_conflict = True
                    best_zero_k = k

            except Exception as e:
                logger.error(f"k={k} QAOA求解失败: {str(e)}")
                continue

        # -----------------------------------------------------------------
        # QAOA无结果：用理论最小色数重试
        # -----------------------------------------------------------------
        if not k_results:
            logger.warning(f"无有效QAOA结果，用理论最小色数={theoretical_min_k}重试")
            try:
                qaoa_result = solve_k_coloring(
                    graph=new_sub,
                    k=theoretical_min_k,
                    p=p,
                    num_steps=num_steps * 2,
                    vertex_colors=vertex_colors,
                    nodes_to_recolor=nodes_to_recolor,
                    penalty=penalty,
                    Q=Q,
                    learning_rate=learning_rate * 0.5,
                    early_stop_threshold=3
                )

                if qaoa_result and len(qaoa_result) == 5:
                    _, conv_param, qaoa_coloring, _, _ = qaoa_result
                    if isinstance(qaoa_coloring, dict) and qaoa_coloring:
                        current_conflict = count_conflicts(qaoa_coloring, new_sub)
                        k_results.append((theoretical_min_k, conv_param, current_conflict, qaoa_coloring))
            except Exception as e:
                logger.error(f"理论最小色数重试失败: {str(e)}")

        # -----------------------------------------------------------------
        # QAOA完全失败：严格限制贪心兜底（仅对简单图）
        # -----------------------------------------------------------------
        # 只有当子图是简单图（边数<3）且QAOA完全失败时才使用贪心
        # 复杂图即使QAOA失败也尽量不使用贪心，而是使用DSATUR作为fallback
        if not k_results:
            if is_complex_subgraph:
                logger.warning(f"复杂图QAOA完全失败，使用DSATUR策略（而非简单贪心）")
                # 使用DSATUR算法（比简单贪心更强）
                dsatur_coloring = nx.coloring.greedy_color(new_sub, strategy='DSATUR')
                required_k = max(dsatur_coloring.values()) + 1 if dsatur_coloring else 2
                current_conflict = count_conflicts(dsatur_coloring, new_sub)
                k_results.append((required_k, 0.0, current_conflict, dsatur_coloring))
            else:
                logger.warning(f"简单图QAOA完全失败，用贪心着色兜底")
                sorted_nodes = sorted(new_sub.nodes, key=lambda x: new_sub.degree(x), reverse=True)
                greedy_coloring, required_k = assign_colors_in_order(new_sub, sorted_nodes, k=2)
                current_conflict = count_conflicts(greedy_coloring, new_sub)
                k_results.append((required_k, 0.0, current_conflict, greedy_coloring))

        # -----------------------------------------------------------------
        # 选择最优k值
        # -----------------------------------------------------------------
        # 按收敛参数分组，选择收敛参数最小的组
        k_results_with_round = [
            (k, round(conv_param, 4), conv_param, conflict, coloring)
            for k, conv_param, conflict, coloring in k_results
        ]
        conv_groups = defaultdict(list)
        for item in k_results_with_round:
            k, round_conv, orig_conv, conflict, coloring = item
            conv_groups[round_conv].append((conflict, k, orig_conv, coloring))

        min_round_conv = min(conv_groups.keys())
        best_group = conv_groups[min_round_conv]
        best_group_sorted = sorted(best_group, key=lambda x: (x[0], x[1]))
        best_conflict, best_k, best_conv, best_coloring = best_group_sorted[0]
        logger.info(f"最优k值选择：k={best_k}（收敛参数={best_conv:.4f}，冲突数={best_conflict}）")

        # -----------------------------------------------------------------
        # 固定k值重试着色（贪心策略）
        # -----------------------------------------------------------------
        sorted_nodes = sorted(new_sub.nodes, key=lambda x: new_sub.degree(x), reverse=True)
        validated_k = validate_min_k(new_sub, best_k)
        fixed_k = max(best_k, validated_k)
        max_retry = 5
        retry_count = 0
        final_coloring = None
        final_conflict = float('inf')
        retry_info = sub_result['retry_info'].copy()

        while retry_count < max_retry:
            retry_count += 1
            retry_info['total_attempts'] = retry_count

            # 首次按度排序，后续打乱非最大度节点顺序
            if retry_count > 1:
                non_max_nodes = sorted_nodes[1:]
                np.random.shuffle(non_max_nodes)
                current_sorted = [sorted_nodes[0]] + non_max_nodes
            else:
                current_sorted = sorted_nodes

            try:
                temp_coloring, required_k = assign_colors_in_order(
                    graph=new_sub,
                    ordered_nodes=current_sorted,
                    k=fixed_k,
                    vertex_colors=vertex_colors
                )
                if required_k > fixed_k:
                    raise ValueError(f"k={fixed_k}不足，需至少{required_k}")

                temp_conflict = count_conflicts(temp_coloring, new_sub)
                retry_info['conflict_history'].append(temp_conflict)
                logger.info(f"第{retry_count}次重试：冲突数={temp_conflict}")

                if temp_conflict < final_conflict:
                    final_conflict = temp_conflict
                    final_coloring = temp_coloring.copy()
                    if temp_conflict == 0 and retry_info['success_attempt'] is None:
                        retry_info['success_attempt'] = retry_count

                if final_conflict == 0:
                    logger.info(f"第{retry_count}次重试找到无冲突方案，提前退出")
                    break

            except Exception as e:
                logger.error(f"第{retry_count}次重试失败: {str(e)}")
                retry_info['conflict_history'].append(float('inf'))
                continue

        # -----------------------------------------------------------------
        # 处理重试结果
        # -----------------------------------------------------------------
        if final_coloring is None or not isinstance(final_coloring, dict):
            final_coloring = best_coloring.copy()
            final_conflict = count_conflicts(final_coloring, new_sub)
            logger.warning(f"所有重试未获有效结果，fallback到QAOA原始方案：冲突数={final_conflict}")

        # 映射回原始节点ID
        final_k = max(final_coloring.values()) + 1 if final_coloring else 1
        reverse_mapping = {new: old for old, new in mapping.items()}
        original_coloring = {
            old: final_coloring[new]
            for old, new in mapping.items()
            if new in final_coloring
        }

        # 确定状态
        if final_conflict == 0:
            sub_result['status'] = 'success'
        elif final_conflict <= len(new_sub.edges) * 0.1:
            sub_result['status'] = 'warning (minimal conflict)'
        else:
            sub_result['status'] = 'failed (high conflict)'

        # 更新子图结果
        sub_result.update({
            'min_k': final_k,
            'coloring': original_coloring,
            'conflict_count': final_conflict,
            'retry_info': retry_info
        })

        # 更新缓存
        if sub_signature not in signature_cache:
            max_color = max(final_coloring.values()) if final_coloring else 0
            signature_cache[sub_signature] = (final_k, final_coloring, max_color)
            logger.info(f"子图 {i + 1} 签名缓存已更新")

        # 记录结果
        processing_time = time.time() - sub_start_time
        results[i] = (
            sub_result['min_k'],
            sub_result['coloring'],
            sub_result['conflict_count'],
            sub_result['status'],
            sub_result['retry_info']
        )
        processed_subgraphs.append(i)
        logger.info(f"子图 {i + 1} 完成：k={final_k}，冲突数={final_conflict}，状态={sub_result['status']}")


    # ======================================================================
    # 2.6 记录子图级日志到主日志
    # ======================================================================
    for i, (k, coloring, conflicts, status, _) in enumerate(results):
        if coloring is None:
            continue
        logger.info(f"{dataset_name},{i},{len(subgraphs[i].nodes)},"
                    f"{len(subgraphs[i].edges)},{k},{conflicts},{status}")

    return results


# ==============================================================================
# 3. 迭代优化函数
# ==============================================================================

_iterative_cache = defaultdict(dict)


def iterative_optimization(
    graph,
    subgraphs,
    sub_mappings,
    subgraph_results=None,
    max_k=10,
    p=1,
    num_steps=1000,
    max_iter=10,
    adjacency_threshold=0.3,
    early_stop_threshold=2,
    penalty= 1000,
    Q=20,
    learning_rate=0.01,
    vertex_colors=None,
    nodes_to_recolor=None,
    dataset_name=None,
    graph_id=None
):
    """
    迭代优化全局着色方案

    优化策略：
    1. 从子图结果构建全局初始着色
    2. 冲突=0时立即返回
    3. 冲突≤1时跳过子图重优化，仅局部微调
    4. 双重早停机制：冲突绝对门限 + 连续无改进轮次

    参数:
        graph: 原始图
        subgraphs: 子图列表
        sub_mappings: 子图节点映射列表
        subgraph_results: 子图处理结果
        max_k: 最大颜色数
        p: QAOA层数
        num_steps: 最大训练步数
        max_iter: 最大迭代次数
        adjacency_threshold: 邻接阈值（未使用）
        early_stop_threshold: 早停阈值
        penalty: 冲突惩罚系数
        Q: 图划分参数
        learning_rate: 学习率
        vertex_colors: 顶点颜色字典
        nodes_to_recolor: 需要重新着色的节点列表
        dataset_name: 数据集名称
        graph_id: 图ID

    返回:
        tuple: (best_coloring, accuracy, conflict_counts, conflict_history, [])
            - best_coloring: 最佳全局着色方案
            - accuracy: 准确率
            - conflict_counts: 每轮迭代冲突数列表
            - conflict_history: 冲突历史记录
            - []: 空列表（保留兼容性）
    """

    # ======================================================================
    # 3.1 参数校验
    # ======================================================================
    if not dataset_name or graph_id is None:
        raise ValueError("必须提供 dataset_name 和 graph_id")

    # ======================================================================
    # 3.2 初始化缓存 & 最佳 k
    # ======================================================================
    subgraph_cache = {}
    best_k = 2

    if subgraph_results:
        for sub, mapping, (k, coloring, _, _, _) in zip(subgraphs, sub_mappings, subgraph_results):
            if isinstance(coloring, dict) and coloring:
                sig = get_graph_signature(sub)
                subgraph_cache[sig] = (k, coloring, max(coloring.values()))
                best_k = max(best_k, k)

    # ======================================================================
    # 3.3 全局着色（复用缓存 / 贪心）
    # ======================================================================
    global_coloring = {}

    for sub, mapping in zip(subgraphs, sub_mappings):
        if len(sub.nodes) == 0:
            continue

        sig = get_graph_signature(sub)
        if sig in subgraph_cache:
            k, coloring, _ = subgraph_cache[sig]
        else:
            k = max(best_k, 2)
            coloring = nx.coloring.greedy_color(sub, strategy='DSATUR')

        reverse = {new: old for old, new in mapping.items()}
        for new_node, color in coloring.items():
            if new_node in reverse:
                global_coloring[reverse[new_node]] = color % max_k

    # 补孤立节点
    for n in graph.nodes:
        if n not in global_coloring:
            used = {global_coloring[nei] for nei in graph.neighbors(n) if nei in global_coloring}
            color = 0
            while color in used:
                color += 1
            global_coloring[n] = color % max_k

    # ======================================================================
    # 3.4 初始冲突评估
    # ======================================================================
    best_coloring = global_coloring.copy()
    best_conflict = count_conflicts(best_coloring, graph)
    total_edges = graph.number_of_edges()
    conflict_counts = [best_conflict]
    conflict_history = [("初始", best_conflict)]
    print(f"🎯 初始冲突 {best_conflict}/{total_edges}  准确率 {1-best_conflict/total_edges:.3f}")

    # ======================================================================
    # 3.5 早停门限
    # ======================================================================
    EARLY_STOP_ABS = 0
    no_improve = 0

    # ======================================================================
    # 3.6 迭代优化
    # ======================================================================
    for it in range(max_iter):
        # 冲突为0，提前结束
        if best_conflict <= EARLY_STOP_ABS:
            print("✅ 冲突已归零，提前结束")
            break

        curr_coloring = best_coloring.copy()
        conflict_edges = find_conflict_edges(curr_coloring, graph)
        if not conflict_edges:
            break

        print(f"\n===== 迭代 {it+1}/{max_iter}  冲突边 {len(conflict_edges)} =====")

        # -------------------------------------------------------------
        # 3.6.1 冲突≤1 → 跳过子图重优化，仅局部微调
        # -------------------------------------------------------------
        if len(conflict_edges) <= 1:
            print("⚙️  单冲突边，跳过子图重优化")
            nodes_to_fix = {n for e in conflict_edges for n in e}
        else:
            # 抽取冲突诱导子图
            conflict_nodes = set(n for e in conflict_edges for n in e)
            extended = conflict_nodes | {nei for n in conflict_nodes for nei in graph.neighbors(n)}
            subG = graph.subgraph(extended).copy()
            mapping = {old: idx for idx, old in enumerate(subG.nodes)}
            rev_map = {idx: old for old, idx in mapping.items()}
            renamed_subG = nx.relabel_nodes(subG, mapping)

            # 子图重优化（使用 DSATUR 快速重着色）
            new_coloring = nx.coloring.greedy_color(renamed_subG, strategy='DSATUR')

            # 映射回原图
            for new_node, color in new_coloring.items():
                curr_coloring[rev_map[new_node]] = color % max_k

            nodes_to_fix = conflict_nodes

        # -------------------------------------------------------------
        # 3.6.2 局部微调（高冲突节点优先）
        # -------------------------------------------------------------
        # 计算冲突节点分数（冲突次数 + 节点度）
        conflict_nodes = {n for e in find_conflict_edges(curr_coloring, graph) for n in e}
        node_score = {n: 0 for n in conflict_nodes}

        for u, v in find_conflict_edges(curr_coloring, graph):
            node_score[u] += 1
            node_score[v] += 1
        for n in conflict_nodes:
            node_score[n] += graph.degree(n)

        # 按分数降序重新着色冲突节点
        for node in sorted(conflict_nodes, key=lambda x: node_score[x], reverse=True):
            neighbors = list(graph.neighbors(node))
            used = {curr_coloring[nei] for nei in neighbors if nei in curr_coloring}
            # 选最小可用颜色
            color = 0
            while color in used:
                color += 1
            new_color = color % max_k
            old_color = curr_coloring[node]
            if new_color != old_color:
                curr_coloring[node] = new_color

        # -------------------------------------------------------------
        # 3.6.3 评估改进
        # -------------------------------------------------------------
        new_conflict = count_conflicts(curr_coloring, graph)
        conflict_counts.append(new_conflict)
        conflict_history.append((f"迭代{it+1}", new_conflict))

        if new_conflict < best_conflict:
            best_conflict = new_conflict
            best_coloring = curr_coloring.copy()
            no_improve = 0
            print(f"✨ 冲突下降 {best_conflict}  准确率 {1-best_conflict/total_edges:.3f}")
        else:
            no_improve += 1
            if no_improve >= early_stop_threshold:
                print(f"⏹️  连续 {early_stop_threshold} 次无改进，早停")
                break

    # ======================================================================
    # 3.7 返回结果
    # ======================================================================
    accuracy = 1 - best_conflict / total_edges if total_edges else 1.0
    print(
        f"\n===== 完成  冲突 {best_conflict}  准确率 {accuracy:.3f}  颜色 {len(set(best_coloring.values()))} =====")

    return best_coloring, accuracy, conflict_counts, conflict_history, []