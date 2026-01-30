"""
标准QAOA多层次图着色模块（天衍平台适配版 - 改进版）

本模块实现基于标准QAOA(Standard QAOA)的多层次图着色算法，主要包含：
1. sequential_process_subgraphs_tianyan: 使用 solve_k_coloring_tianyan_with_training 求解子图着色
2. iterative_optimization_tianyan: 迭代优化全局着色方案

改进点：
- 使用 solve_k_coloring_tianyan_with_training 集成训练和执行
- 支持热启动机制（参考 solve_k_coloring_standard）
- 支持参数缓存和同构图参数复用
- 改进的早停机制和能量监控

算法特点：
- 集成训练：训练和执行统一在 solve_k_coloring_tianyan_with_training 中
- 热启动：逐步增加k值时使用上一轮的最优参数
- 标准QAOA：使用训练好的参数在天衍真机上执行
- 早停机制：基于冲突数变化和迭代次数
"""

# ==============================================================================
# 导入模块
# ==============================================================================

import csv
import time
import traceback
import json
import os
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# cqlib 用于天衍真机执行
from cqlib import TianYanPlatform
import numpy as np
import networkx as nx
import matplotlib
# 使用非交互式后端，图片显示后不阻塞程序继续执行
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 从共享模块导入通用函数
from multilevel_common_tianyan import (
    count_conflicts,
    extract_coloring_tianyan,
    is_complete_graph,
    is_cycle_graph,
    is_chain_graph,
    chain_graph_coloring,
    get_graph_signature,
    setup_logger,
    cycle_graph_coloring,
    assign_colors_in_order,
    validate_min_k,
    find_conflict_edges,
    _greedy_coloring_from_max_degree,
    _resolve_conflicts_with_greedy,
    get_subgraph_coloring,
    visualize_graph,
    solve_k_coloring_tianyan_with_training
)


# ==============================================================================
# 1. 改进的顺序子图处理函数（天衍平台版本）
# ==============================================================================

def sequential_process_subgraphs_tianyan(
    subgraphs: List[nx.Graph],
    sub_mappings: List[Dict],
    dataset_name: str,
    graph_id: int,
    platform: TianYanPlatform,
    lab_id: Optional[str],
    max_k: int = 20,
    p: int = 1,
    num_shots: int = 1000,
    train_params: bool = True,
    train_max_iter: int = 50,
    train_lr: float = 0.01,
    train_num_shots: int = 100,
    early_stop_threshold: int = 10,
    algorithm: str = 'standard',
    graph_name: Optional[str] = None,
    stop_on_zero_conflict: bool = False,
    max_qubits: int = 66
) -> List[Tuple[int, Dict, int, str, Dict]]:
    """
    顺序子图处理函数（使用 solve_k_coloring_tianyan_with_training）

    改进点：
    1. 集成训练模式：使用 solve_k_coloring_tianyan_with_training
    2. 支持参数训练（参考 solve_k_coloring_standard 的训练逻辑）
    3. 支持热启动机制
    4. 记录训练过程的能量历史
    5. 支持早停机制（找到无冲突解后立即停止训练）

    参数:
        subgraphs: 子图列表
        sub_mappings: 子图节点映射列表
        dataset_name: 数据集名称
        graph_id: 图ID
        platform: TianYanPlatform 实例
        lab_id: 实验室ID
        max_k: 最大尝试的颜色数
        p: QAOA层数
        num_shots: 最终采样次数
        train_params: 是否训练参数
        train_max_iter: 训练最大迭代次数
        train_lr: 训练学习率
        train_num_shots: 训练时的采样次数（小样本）
        early_stop_threshold: 早停阈值
        algorithm: 算法类型 ('standard')
        graph_name: 数据图名称（用于天衍实验命名）
        stop_on_zero_conflict: 找到无冲突解后是否立即停止训练（优化项）
        max_qubits: 量子比特数限制（根据机时包限制调整）

    返回:
        list: 子图结果列表，每个元素为 (min_k, coloring, conflict_count, status, retry_info)
    """
    # ======================================================================
    # 初始化
    # ======================================================================
    logger = setup_logger(dataset_name, str(graph_id))
    results = [None] * len(subgraphs)

    # ======================================================================
    # 批量处理孤立节点
    # ======================================================================
    isolated_subgraphs = [
        (i, sub, mapping)
        for i, (sub, mapping) in enumerate(zip(subgraphs, sub_mappings))
        if len(sub.nodes) > 0 and len(sub.edges) == 0
    ]
    
    if isolated_subgraphs:
        logger.info(f"\n===== 开始处理 {len(isolated_subgraphs)} 个孤立节点子图 =====")
        
        for i, sub, mapping in isolated_subgraphs:
            try:
                new_sub = nx.relabel_nodes(sub, mapping)
                global_coloring = {node: 0 for node in new_sub.nodes}
                original_coloring = {
                    old: global_coloring[new]
                    for old, new in mapping.items()
                    if new in global_coloring
                }
                
                results[i] = (
                    1,
                    original_coloring,
                    0,
                    'success (isolated nodes)',
                    {'batch_processed': True, 'total_attempts': 1, 'trained': False}
                )
                
                logger.info(f"孤立子图 {i+1} 完成：节点数={len(sub.nodes)}，k=1，冲突数=0")
                
            except Exception as e:
                logger.error(f"孤立子图 {i+1} 处理失败: {str(e)}")
                results[i] = (1, {}, float('inf'), 'failed', {'error': str(e)})
    
    # ======================================================================
    # 主循环：处理非孤立子图
    # ======================================================================
    for i, (sub, mapping) in enumerate(zip(subgraphs, sub_mappings)):
        if results[i] is not None:
            continue

        sub_start_time = time.time()

        # ---------------------------------------------------------
        # 空子图处理
        # ---------------------------------------------------------
        if len(sub.nodes) == 0:
            results[i] = (1, {}, 0, 'skipped (empty subgraph)', {})
            logger.info(f"子图 {i+1}: 空子图，跳过")
            continue

        logger.info(f"\n===== 处理子图 {i+1}：节点数={len(sub.nodes)}，边数={len(sub.edges)} =====")

        try:
            new_sub = nx.relabel_nodes(sub, mapping)

            # ---------------------------------------------------------
            # 节点映射后再次检查
            # ---------------------------------------------------------
            if len(new_sub.nodes) == 0:
                results[i] = (1, {}, 0, 'skipped (empty after mapping)', {})
                logger.info(f"子图 {i+1}: 映射后为空子图，跳过")
                continue
            
            # ---------------------------------------------------------
            # 提取子图内部的孤立节点（degree=0）并单独着色
            # ---------------------------------------------------------
            isolated_nodes_in_sub = [node for node in new_sub.nodes if new_sub.degree(node) == 0]
            if isolated_nodes_in_sub:
                logger.info(f"\n___________子图 {i+1} 包含 {len(isolated_nodes_in_sub)} 个孤立节点，单独着色____________")
                
                # 创建不含孤立节点的子图用于后续处理
                non_isolated_nodes = [node for node in new_sub.nodes if node not in isolated_nodes_in_sub]
                if len(non_isolated_nodes) > 0:
                    new_sub_without_isolated = new_sub.subgraph(non_isolated_nodes).copy()
                else:
                    # 如果所有节点都是孤立的，直接返回1色着色
                    original_coloring = {
                        old: 0
                        for old, new in mapping.items()
                        if new in new_sub.nodes
                    }
                    results[i] = (
                        1,
                        original_coloring,
                        0,
                        'success (all isolated nodes)',
                        {'trained': False, 'total_attempts': 1}
                    )
                    logger.info(f"子图 {i+1}: 所有节点都是孤立的，k=1，冲突数=0")
                    continue
                
                logger.info(f"  原始节点数={len(new_sub.nodes)}，孤立节点数={len(isolated_nodes_in_sub)}，剩余节点数={len(non_isolated_nodes)}")
            else:
                new_sub_without_isolated = new_sub
                isolated_nodes_in_sub = []
            
            # ---------------------------------------------------------
            # 链式图专用处理（QAOA不参与）- 对不含孤立节点的子图
            # ---------------------------------------------------------
            if is_chain_graph(new_sub_without_isolated):
                n = new_sub_without_isolated.number_of_nodes()
                logger.info(f"\n___________处理子图 {i+1}（链式图，节点数：{n}）____________")
                logger.info(f"检测到链式图，使用2色着色（QAOA不参与）")

                try:
                    # 使用链式图专用着色算法（使用2种颜色）
                    chain_coloring, chain_k = chain_graph_coloring(new_sub_without_isolated)
                    conflict_count = count_conflicts(chain_coloring, new_sub_without_isolated)
                    if conflict_count != 0:
                        raise RuntimeError(f"链式图着色异常！预期0冲突，实际{conflict_count}冲突")

                    # 添加孤立节点的着色（颜色0）
                    for isolated_node in isolated_nodes_in_sub:
                        chain_coloring[isolated_node] = 0

                    # 映射回原始节点ID
                    reverse_mapping = {new: old for old, new in mapping.items()}
                    original_coloring = {}
                    for new_node, color in chain_coloring.items():
                        if new_node in reverse_mapping:
                            original_coloring[reverse_mapping[new_node]] = color

                    results[i] = (
                        chain_k,
                        original_coloring,
                        conflict_count,
                        'success (chain graph)',
                        {'trained': False, 'total_attempts': 1, 'conflict_history': [0]}
                    )

                    processing_time = time.time() - sub_start_time
                    logger.info(f"子图 {i+1} 完成：k={chain_k}，冲突数=0，状态=success (chain graph)")
                    continue

                except Exception as e:
                    logger.warning(f"链式图着色失败: {str(e)}，切换贪心着色兜底")
                    # 贪心着色兜底
                    reverse_mapping = {new: old for old, new in mapping.items()}
                    sorted_nodes = sorted(new_sub_without_isolated.nodes, key=lambda x: new_sub_without_isolated.degree(x), reverse=True)
                    greedy_coloring, required_k = assign_colors_in_order(
                        new_sub_without_isolated, sorted_nodes, k=2
                    )
                    # 添加孤立节点的着色
                    for isolated_node in isolated_nodes_in_sub:
                        greedy_coloring[isolated_node] = 0
                    conflict_count = count_conflicts(greedy_coloring, new_sub)
                    original_coloring = {
                        reverse_mapping[new]: greedy_coloring[new]
                        for new in greedy_coloring
                        if new in reverse_mapping
                    }

                    results[i] = (
                        required_k,
                        original_coloring,
                        conflict_count,
                        'success (chain fallback greedy)',
                        {'trained': False, 'total_attempts': 1, 'conflict_history': [conflict_count]}
                    )

                    processing_time = time.time() - sub_start_time
                    logger.info(f"子图 {i+1} 兜底完成：k={required_k}，冲突数={conflict_count}")
                    continue

            # ---------------------------------------------------------
            # 环图专用处理（QAOA不参与）
            # ---------------------------------------------------------
            if is_cycle_graph(new_sub_without_isolated):
                n = new_sub_without_isolated.number_of_nodes()
                cycle_type = "Even_ring" if n % 2 == 0 else "Odd_ring"
                logger.info(f"\n___________处理子图 {i+1}（{cycle_type}，节点数：{n}）____________")
                logger.info(f"检测到{cycle_type}，使用环图专用着色（QAOA不参与）")

                try:
                    # 使用环图专用着色算法
                    cycle_coloring, cycle_k = cycle_graph_coloring(new_sub_without_isolated)
                    conflict_count = count_conflicts(cycle_coloring, new_sub_without_isolated)
                    if conflict_count != 0:
                        raise RuntimeError(f"环图着色异常！预期0冲突，实际{conflict_count}冲突")

                    # 添加孤立节点的着色（颜色0）
                    for isolated_node in isolated_nodes_in_sub:
                        cycle_coloring[isolated_node] = 0

                    # 映射回原始节点ID
                    reverse_mapping = {new: old for old, new in mapping.items()}
                    original_coloring = {}
                    for new_node, color in cycle_coloring.items():
                        if new_node in reverse_mapping:
                            original_coloring[reverse_mapping[new_node]] = color

                    results[i] = (
                        cycle_k,
                        original_coloring,
                        conflict_count,
                        'success (cycle graph)',
                        {'trained': False, 'total_attempts': 1, 'conflict_history': [0]}
                    )

                    processing_time = time.time() - sub_start_time
                    logger.info(f"子图 {i+1} 完成：k={cycle_k}，冲突数=0，状态=success (cycle graph)")
                    continue

                except Exception as e:
                    logger.warning(f"环图着色失败: {str(e)}，切换贪心着色兜底")
                    # 贪心着色兜底
                    reverse_mapping = {new: old for old, new in mapping.items()}
                    n_non_isolated = len(new_sub_without_isolated.nodes)
                    sorted_nodes = sorted(new_sub_without_isolated.nodes, key=lambda x: new_sub_without_isolated.degree(x), reverse=True)
                    greedy_coloring, required_k = assign_colors_in_order(
                        new_sub_without_isolated, sorted_nodes, k=3 if n_non_isolated % 2 else 2
                    )
                    # 添加孤立节点的着色
                    for isolated_node in isolated_nodes_in_sub:
                        greedy_coloring[isolated_node] = 0
                    conflict_count = count_conflicts(greedy_coloring, new_sub)
                    original_coloring = {
                        reverse_mapping[new]: greedy_coloring[new]
                        for new in greedy_coloring
                        if new in reverse_mapping
                    }

                    results[i] = (
                        required_k,
                        original_coloring,
                        conflict_count,
                        'success (cycle fallback greedy)',
                        {'trained': False, 'total_attempts': 1, 'conflict_history': [conflict_count]}
                    )

                    processing_time = time.time() - sub_start_time
                    logger.info(f"子图 {i+1} 兜底完成：k={required_k}，冲突数={conflict_count}")
                    continue
            
            # ---------------------------------------------------------
            # 普通图：集成训练 + 天衍真机执行
            # ---------------------------------------------------------
            # 计算理论最小k值（使用不含孤立节点的子图）
            n_non_isolated = new_sub_without_isolated.number_of_nodes()
            max_degree = max(new_sub_without_isolated.degree(node) for node in new_sub_without_isolated.nodes) if n_non_isolated > 1 else 0
            is_complete = is_complete_graph(new_sub_without_isolated)

            theoretical_min_k = n_non_isolated if is_complete else max_degree
            max_test_k = min(theoretical_min_k, max_k)
            max_test_k = max(max_test_k, 2)

            logger.info(f"理论最小k={theoretical_min_k}，测试范围[2, {max_test_k}]")

            # 如果所有节点都是孤立的，直接返回
            if n_non_isolated == 0:
                # 所有节点都是孤立的，使用1色着色
                original_coloring = {
                    old: 0
                    for old, new in mapping.items()
                    if new in new_sub.nodes
                }
                results[i] = (
                    1,
                    original_coloring,
                    0,
                    'success (all isolated nodes)',
                    {'trained': False, 'total_attempts': 1}
                )
                logger.info(f"子图 {i+1}: 所有节点都是孤立的，k=1，冲突数=0")
                continue

            # 使用顺序增加k值策略寻找最小可行k值
            k_results = []
            prev_k_params = None  # 热启动参数
            found_zero_conflict = False
            best_k_val = None
            best_coloring = None
            best_conflict = float('inf')
            best_query_id = None
            best_params = None
            candidate_k_values = []  # 记录尝试过的k值

            logger.info(f"使用顺序增加k值策略寻找最小可行k值，范围[2, {max_test_k}]")

            # 从k=2开始，逐步增加到max_test_k
            for current_k in range(2, max_test_k + 1):
                candidate_k_values.append(current_k)
                logger.info(f"\n--- 尝试 k={current_k} (总范围: [2, {max_test_k}]) ---")

                # 使用 solve_k_coloring_tianyan_with_training（对不含孤立节点的子图）
                logger.info(f"使用训练模式 (max_iter={train_max_iter}, lr={train_lr})")

                try:
                    k_val, coloring, conflict_count, query_id, training_result = solve_k_coloring_tianyan_with_training(
                        platform=platform,
                        graph=new_sub_without_isolated,
                        k=current_k,
                        p=p,
                        prev_params=prev_k_params,  # 热启动
                        lab_id=lab_id,
                        num_shots=num_shots,
                        num_steps=train_max_iter,
                        learning_rate=train_lr,
                        early_stop_threshold=early_stop_threshold,
                        train_num_shots=train_num_shots,
                        algorithm=algorithm,
                        verbose=False,
                        logger=logger,
                        graph_name=graph_name,
                        stop_on_zero_conflict=stop_on_zero_conflict,
                        max_qubits=max_qubits
                    )

                    if coloring:
                        # 保存当前k的最优参数用于下一轮热启动
                        current_params = training_result.get('best_params', {})
                        if current_params:  # 只有当有有效参数时才更新热启动参数
                            prev_k_params = current_params

                        # 保存结果（注意：这里保存的是不含孤立节点的着色）
                        k_results.append((current_k, conflict_count, coloring, query_id, current_params))
                        logger.info(f"k={current_k} 完成：冲突数={conflict_count}，"
                                  f"训练迭代={training_result.get('iterations', 0)}")

                        # 更新最优结果
                        if conflict_count < best_conflict:
                            best_conflict = conflict_count
                            best_k_val = current_k
                            best_coloring = coloring
                            best_query_id = query_id
                            best_params = current_params

                        if conflict_count == 0:
                            logger.info(f"找到无冲突方案，停止尝试更大的k值")
                            found_zero_conflict = True
                            best_k_val = current_k
                            best_coloring = coloring
                            best_query_id = query_id
                            best_params = current_params
                            # 找到无冲突解，停止尝试
                            break
                    else:
                        logger.warning(f"k={current_k} 训练完成但返回空着色，继续尝试更大的k值")

                except Exception as e:
                    logger.error(f"k={current_k} 训练执行失败: {str(e)}")
                    traceback.print_exc()
                    # 执行失败，继续尝试下一个更大的k值
                    continue

            # 如果顺序搜索没有找到无冲突解，检查所有尝试过的k值结果
            if not found_zero_conflict and k_results:
                logger.info("顺序搜索未找到无冲突解，从所有尝试过的k值中选择最佳结果")
                # 从已尝试的k值中选择最佳结果
                k_results_sorted = sorted(k_results, key=lambda x: (x[1], x[0]))
                best_k_val, best_conflict, best_coloring, best_query_id, best_params = k_results_sorted[0]

            # ---------------------------------------------------------
            # 处理结果
            # ---------------------------------------------------------
            if not k_results:
                # QAOA无结果，贪心兜底（使用不含孤立节点的子图）
                logger.warning("QAOA无有效结果，使用贪心着色兜底")
                logger.info(f"  测试的k值: {candidate_k_values}")
                logger.info(f"  原因: 所有k值训练完成后都返回空着色")

                sorted_nodes = sorted(new_sub_without_isolated.nodes, key=lambda x: new_sub_without_isolated.degree(x), reverse=True)
                greedy_coloring, required_k = assign_colors_in_order(
                    new_sub_without_isolated, sorted_nodes, k=2
                )
                # 添加孤立节点的着色
                for isolated_node in isolated_nodes_in_sub:
                    greedy_coloring[isolated_node] = 0
                conflict_count = count_conflicts(greedy_coloring, new_sub)

                reverse_mapping = {new: old for old, new in mapping.items()}
                original_coloring = {
                    reverse_mapping[new]: greedy_coloring[new]
                    for new in greedy_coloring
                    if new in reverse_mapping
                }

                results[i] = (
                    required_k,
                    original_coloring,
                    conflict_count,
                    'success (greedy fallback)',
                    {'trained': False}
                )

                logger.info(f"子图 {i+1} 兜底完成：k={required_k}，冲突数={conflict_count}")

            else:
                # 使用顺序搜索找到的最佳结果
                if best_k_val is not None and best_coloring is not None:
                    if found_zero_conflict:
                        logger.info(f"  顺序搜索找到的无冲突解: k={best_k_val}, 冲突数={best_conflict}")
                        logger.info(f"  尝试过的k值: {sorted(candidate_k_values)}")
                    else:
                        logger.info(f"  顺序搜索未找到无冲突解，选择冲突数最小的结果: k={best_k_val}, 冲突数={best_conflict}")
                        logger.info(f"  尝试过的k值: {sorted(candidate_k_values)}")
                else:
                    # 如果顺序搜索没有找到有效结果，从k_results中选择
                    k_results_sorted = sorted(k_results, key=lambda x: (x[1], x[0]))
                    best_k_val, best_conflict, best_coloring, best_query_id, best_params = k_results_sorted[0]
                    logger.info(f"  从所有尝试结果中选择最佳: k={best_k_val}, 冲突数={best_conflict}")
                    logger.info(f"  尝试过的k值: {sorted(candidate_k_values)}")

                # 添加孤立节点的着色（颜色0）
                for isolated_node in isolated_nodes_in_sub:
                    best_coloring[isolated_node] = 0

                reverse_mapping = {new: old for old, new in mapping.items()}
                original_coloring = {
                    reverse_mapping[new]: best_coloring[new]
                    for new in best_coloring
                    if new in reverse_mapping
                }

                # 确定状态
                if best_conflict == 0:
                    status = 'success (with training)' if best_params else 'success (without training)'
                elif best_conflict <= len(new_sub_without_isolated.edges) * 0.1:
                    status = 'warning (minimal conflict)'
                else:
                    status = 'failed (high conflict)'

                results[i] = (
                    best_k_val,
                    original_coloring,
                    best_conflict,
                    status,
                    {
                        'query_id': best_query_id,
                        'trained': best_params is not None,
                        'trained_params': best_params
                    }
                )

                logger.info(f"子图 {i+1} 完成：k={best_k_val}，冲突数={best_conflict}，状态={status}")
            
        except Exception as e:
            logger.error(f"子图 {i+1} 处理失败: {str(e)}")
            traceback.print_exc()
            results[i] = (1, {}, float('inf'), 'failed', {'error': str(e)})
    
    # ======================================================================
    # 记录子图级日志
    # ======================================================================
    for i, (k, coloring, conflicts, status, _) in enumerate(results):
        if coloring is not None:
            logger.info(f"{dataset_name},{i},{len(subgraphs[i].nodes)},"
                       f"{len(subgraphs[i].edges)},{k},{conflicts},{status}")
    
    return results


# ==============================================================================
# 3. 改进的迭代优化函数
# ==============================================================================

def iterative_optimization_tianyan(
    graph: nx.Graph,
    subgraphs: List[nx.Graph],
    sub_mappings: List[Dict],
    subgraph_results: Optional[List] = None,
    max_k: int = 10,
    p: int = 1,
    max_iter: int = 10,
    early_stop_threshold: int = 2,
    dataset_name: Optional[str] = None,
    graph_id: Optional[int] = None
) -> Tuple[Dict, float, List, List, Dict]:
    """
    改进的迭代优化全局着色方案
    
    改进点：
    1. 更详细的优化日志
    2. 支持从子图结果中提取训练参数信息
    3. 可视化优化过程
    
    参数:
        graph: 原始图
        subgraphs: 子图列表
        sub_mappings: 子图节点映射列表
        subgraph_results: 子图处理结果
        max_k: 最大颜色数
        p: QAOA层数（保留兼容性）
        max_iter: 最大迭代次数
        early_stop_threshold: 早停阈值
        dataset_name: 数据集名称
        graph_id: 图ID
    
    返回:
        tuple: (best_coloring, accuracy, conflict_counts, conflict_history, training_info)
    """
    if not dataset_name or graph_id is None:
        raise ValueError("必须提供 dataset_name 和 graph_id")
    
    logger = setup_logger(dataset_name, str(graph_id))
    
    # ======================================================================
    # 初始化
    # ======================================================================
    subgraph_cache = {}
    best_k = 2
    training_info = {}
    
    # 从子图结果构建缓存
    if subgraph_results:
        for i, (sub, mapping, (k, coloring, _, _, retry_info)) in enumerate(
            zip(subgraphs, sub_mappings, subgraph_results)
        ):
            if isinstance(coloring, dict) and coloring:
                sig = get_graph_signature(sub)
                subgraph_cache[sig] = (k, coloring, max(coloring.values()))
                best_k = max(best_k, k)
                
                # 记录训练信息
                if retry_info and retry_info.get('trained', False):
                    training_info[f'subgraph_{i}'] = {
                        'k': k,
                        'params': retry_info.get('trained_params'),
                        'trained': True
                    }
    
    # ======================================================================
    # 全局着色初始化
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
        # 检查是否为完全图的完美着色（冲突数为0且每个节点颜色唯一）
        is_perfect_coloring = (len(set(coloring.values())) == len(sub.nodes) and
                               all(color < len(sub.nodes) for color in coloring.values()))

        for new_node, color in coloring.items():
            if new_node in reverse:
                # 如果是完美着色且颜色在合理范围内，直接使用原始颜色
                if is_perfect_coloring and color < max_k:
                    global_coloring[reverse[new_node]] = color
                else:
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
    # 迭代优化
    # ======================================================================
    best_coloring = global_coloring.copy()
    best_conflict = count_conflicts(best_coloring, graph)
    total_edges = graph.number_of_edges()
    conflict_counts = [best_conflict]
    conflict_history = [("初始", best_conflict)]
    
    logger.info(f"初始冲突: {best_conflict}/{total_edges}，准确率: {1-best_conflict/total_edges:.3f}")
    
    EARLY_STOP_ABS = 0
    no_improve = 0
    
    for it in range(max_iter):
        if best_conflict <= EARLY_STOP_ABS:
            logger.info("冲突已归零，提前结束")
            break
        
        logger.info(f"\n===== 迭代 {it+1}/{max_iter} =====")
        
        # 冲突边检测
        conflict_edges = find_conflict_edges(best_coloring, graph)
        if not conflict_edges:
            break
        
        logger.info(f"冲突边数: {len(conflict_edges)}")
        
        # 局部优化
        curr_coloring = best_coloring.copy()
        
        # 计算冲突节点分数
        conflict_nodes = {n for e in conflict_edges for n in e}
        node_score = {n: 0 for n in conflict_nodes}
        
        for u, v in conflict_edges:
            node_score[u] += 1
            node_score[v] += 1
        for n in conflict_nodes:
            node_score[n] += graph.degree(n)
        
        # 按分数降序重新着色
        for node in sorted(conflict_nodes, key=lambda x: node_score[x], reverse=True):
            neighbors = list(graph.neighbors(node))
            used = {curr_coloring[nei] for nei in neighbors if nei in curr_coloring}
            
            color = 0
            while color in used:
                color += 1
            new_color = color % max_k
            
            curr_coloring[node] = new_color
        
        # 评估改进
        new_conflict = count_conflicts(curr_coloring, graph)
        conflict_counts.append(new_conflict)
        conflict_history.append((f"迭代{it+1}", new_conflict))
        
        logger.info(f"冲突数: {new_conflict}")
        
        if new_conflict < best_conflict:
            best_conflict = new_conflict
            best_coloring = curr_coloring.copy()
            no_improve = 0
            logger.info(f"✓ 冲突下降: {best_conflict}")
        else:
            no_improve += 1
            if no_improve >= early_stop_threshold:
                logger.info(f"连续{early_stop_threshold}次无改进，早停")
                break
    
    # ======================================================================
    # 返回结果
    # ======================================================================
    accuracy = 1 - best_conflict / total_edges if total_edges else 1.0
    
    result_msg = f"\n===== 优化完成 =====\n"
    result_msg += f"最优冲突: {best_conflict}/{total_edges}\n"
    result_msg += f"准确率: {accuracy:.3f}\n"
    result_msg += f"使用颜色数: {len(set(best_coloring.values()))}\n"
    
    logger.info(result_msg)
    
    return best_coloring, accuracy, conflict_counts, conflict_history, training_info


# ==============================================================================
# 4. 辅助函数：能量可视化
# ==============================================================================

def plot_energy_convergence(
    energy_history: List[float],
    save_path: Optional[str] = None,
    title: str = "QAOA Energy Convergence"
):
    """
    绘制QAOA能量收敛曲线
    
    参数:
        energy_history: 能量历史列表
        save_path: 保存路径（可选）
        title: 图表标题
    """
    if not energy_history:
        print("警告: 能量历史为空，无法绘图")
        return
    
    plt.figure(figsize=(10, 6))
    
    iterations = range(1, len(energy_history) + 1)
    plt.plot(iterations, energy_history, 'b-', linewidth=2, label='Energy')
    
    # 标记最优点
    min_idx = np.argmin(energy_history)
    plt.scatter([min_idx + 1], [energy_history[min_idx]], 
               color='red', s=100, zorder=5, label=f'Best: {energy_history[min_idx]:.6f}')
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"能量收敛图已保存: {save_path}")

    plt.close()
