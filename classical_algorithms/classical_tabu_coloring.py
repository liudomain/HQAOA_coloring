import os
import time
import traceback
import networkx as nx
import numpy as np
from tabucol import tabucol, estimate_chromatic_number
from graph_coloring_utils import GraphColoringUtils
import matplotlib.pyplot as plt

# 初始化工具类
utils = GraphColoringUtils(
    data_dir=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'Data', 'instances'
    ),
    results_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coloring_results')
)




def improved_chromatic_estimate(graph):
    """
    改进的色数估算函数，结合多种下界计算方法：
    1. 最大度数 + 1 的下界（Brooks定理的推论）
    2. 团的大小下界
    3. 边密度调整
    """
    if not graph or graph.number_of_nodes() == 0:
        return 1
    
    # 基础下界1：最大度数 + 1
    max_degree = max(dict(graph.degree()).values())
    lower_bound_1 = max_degree + 1
    
    # 基础下界2：最大团大小（近似计算）
    # 使用贪心算法找最大团的近似值
    nodes = list(graph.nodes())
    max_clique_size = 1
    
    for start_node in nodes[:min(20, len(nodes))]:  # 限制采样数量提高效率
        clique = {start_node}
        candidates = set(graph.neighbors(start_node))
        
        while candidates:
            # 选择度数最高的候选节点
            next_node = max(candidates, key=lambda n: graph.degree(n))
            clique.add(next_node)
            # 更新候选集：必须与当前团中所有节点相连
            candidates = candidates.intersection(set(graph.neighbors(next_node)))
            candidates -= clique
            
        max_clique_size = max(max_clique_size, len(clique))
    
    lower_bound_2 = max_clique_size
    
    # 边密度调整
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    edge_density = m / (n * (n - 1) / 2) if n > 1 else 0
    
    # 根据边密度进行调整
    if edge_density > 0.7:  # 稠密图
        adjustment = max(1, int((edge_density - 0.5) * 10))
    else:
        adjustment = 0
    
    # 综合估算
    estimated_k = max(lower_bound_1, lower_bound_2) + adjustment
    
    return min(estimated_k, n)  # 色数不可能超过节点数





def process_single_graph(filename, graph, max_iterations_base=20000, tabu_size_ratio=0.15, reps=100, max_retries=5):
    """
    改进的Tabu算法图着色处理，专注于最小着色数
    优化策略：
    1. 自适应参数调整
    2. 渐进式色数增长
    3. 多种策略组合
    4. 智能重启机制
    """
    if graph is None:
        print("❌ process_single_graph: 输入图为None，无法处理")
        return None
    n_nodes = graph.number_of_nodes()
    if n_nodes == 0:
        print("❌ process_single_graph: 图中无节点，无法处理")
        return None

    print(f"\n🔍 开始处理图：{filename}")
    print(f"  节点数: {n_nodes}, 边数: {graph.number_of_edges()}, 最大度数: {max(dict(graph.degree()).values())}")

    # 自适应参数：根据图特征动态调整
    max_degree = max(dict(graph.degree()).values()) if graph.nodes() else 0
    edge_density = graph.number_of_edges() / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
    
    # 根据图的特征调整参数
    if edge_density > 0.5:  # 稠密图
        max_iterations = max(max_iterations_base * 2, n_nodes * 200)
        tabu_size = max(10, int(n_nodes * 0.2))
        reps = 150
    elif edge_density > 0.2:  # 中等密度图
        max_iterations = max(max_iterations_base, n_nodes * 150)
        tabu_size = max(7, int(n_nodes * 0.15))
        reps = 100
    else:  # 稀疏图
        max_iterations = max(max_iterations_base // 2, n_nodes * 100)
        tabu_size = max(5, int(n_nodes * 0.1))
        reps = 80

    print(f"  自适应参数：迭代次数={max_iterations}, 禁忌表大小={tabu_size}, 重试次数={reps}")
    print(f"  图特征：边密度={edge_density:.3f}, 最大度数={max_degree}")

    start = time.perf_counter()
    
    # 使用改进的色数估算
    k = improved_chromatic_estimate(graph)
    # 同时保留原估算作为参考
    original_k = estimate_chromatic_number(graph)
    lower_bound = max_degree + 1
    
    # 选择更合理的初始色数
    k = max(k, original_k, lower_bound, 1)
    print(f"  初始色数估算：{k}（改进算法:{improved_chromatic_estimate(graph)}, 原算法:{original_k}, 下界:{lower_bound}）")

    best_coloring = None
    best_num_colors = float('inf')
    best_conflicts = float('inf')
    
    # 渐进式搜索策略
    color_attempts = []
    
    # 策略1：从估算值开始，逐步增加
    for delta in range(0, max_retries):
        current_k = k + delta
        print(f"  尝试颜色数={current_k}（delta={delta}）")
        
        coloring = tabucol(
            graph, current_k,
            tabu_size=tabu_size,
            reps=reps,
            max_iterations=max_iterations,
            debug=False  # 减少输出提高性能
        )
        
        if coloring is not None:
            # 检查覆盖所有节点
            missing_nodes = [node for node in graph.nodes() if node not in coloring]
            if not missing_nodes:
                # 归一化并检查实际使用颜色数
                normalized_coloring, actual_colors = utils.normalize_coloring(coloring)
                conflicts = utils.calculate_conflicts(graph, normalized_coloring)
                
                print(f"    成功！实际使用颜色数：{actual_colors}, 冲突数：{conflicts}")
                
                if conflicts == 0 and actual_colors < best_num_colors:
                    best_coloring = normalized_coloring
                    best_num_colors = actual_colors
                    best_conflicts = conflicts
                    print(f"    🎯 找到更好的解！颜色数：{actual_colors}")
                
                color_attempts.append((actual_colors, conflicts, normalized_coloring))
                
                # 如果找到无冲突解且颜色数合理，可以提前终止
                if conflicts == 0 and actual_colors <= lower_bound + 2:
                    break
            else:
                print(f"    着色方案缺失{len(missing_nodes)}个节点")
        else:
            print(f"    尝试失败")

    # 策略2：如果策略1效果不好，尝试更大的色数
    if best_num_colors == float('inf') or best_conflicts > 0:
        print(f"  策略1未找到理想解，尝试更大的色数范围...")
        for current_k in range(best_num_colors if best_num_colors != float('inf') else k + max_retries, 
                               k + max_retries + 3):
            print(f"  补充尝试颜色数={current_k}")
            coloring = tabucol(graph, current_k, tabu_size=tabu_size, reps=reps, 
                             max_iterations=max_iterations, debug=False)
            
            if coloring:
                normalized_coloring, actual_colors = utils.normalize_coloring(coloring)
                conflicts = utils.calculate_conflicts(graph, normalized_coloring)
                
                if conflicts == 0 and actual_colors < best_num_colors:
                    best_coloring = normalized_coloring
                    best_num_colors = actual_colors
                    best_conflicts = conflicts
                    print(f"    🎯 补充策略找到更好解！颜色数：{actual_colors}")
                    break

    exec_time = (time.perf_counter() - start) * 1000
    
    # 最终结果处理
    if best_coloring is not None:
        coloring = best_coloring
        num_colors = best_num_colors
        conflicts = best_conflicts
        is_valid = (conflicts == 0)
        print(f"  🏆 最终结果：颜色数={num_colors}, 冲突数={conflicts}, 有效={is_valid}")
    else:
        # 使用最后一次尝试的结果
        if color_attempts:
            coloring = color_attempts[-1][2]
            num_colors = color_attempts[-1][0]
            conflicts = color_attempts[-1][1]
            is_valid = (conflicts == 0)
            print(f"  ⚠️ 使用最后尝试结果：颜色数={num_colors}, 冲突数={conflicts}")
        else:
            coloring = {}
            num_colors, conflicts, is_valid = k, -1, False
            print(f"  ❌ 所有尝试均失败")

    print(f"  总耗时：{exec_time:.2f}ms")

    # 使用工具类进行可视化
    if coloring and num_colors > 0:
        utils.visualize_coloring(graph, coloring, filename, num_colors, exec_time, "Tabu")

    return {
        'filename': filename,
        'num_nodes': n_nodes,
        'num_edges': graph.number_of_edges(),
        'num_colors': num_colors,
        'conflicts': conflicts,
        'execution_time_ms': round(exec_time, 2),
        'is_valid': is_valid,
        'coloring': coloring,
        'algorithm': 'Tabu'
    }


def tabu_algorithm_handler(filename, graph):
    """
    Tabu算法处理函数的包装器
    Args:
        filename: str 文件名
        graph: networkx.Graph 图对象
    Returns:
        dict: 处理结果
    """
    # 调用处理函数
    result = process_single_graph(filename, graph)
    
    if result:
        # 标准化结果格式，确保包含必需的字段
        result.setdefault('filename', filename)
        result.setdefault('num_nodes', graph.number_of_nodes())
        result.setdefault('num_edges', graph.number_of_edges())
        result.setdefault('is_valid', result.get('conflicts', 0) == 0)
        
        return result
    return None


if __name__ == "__main__":
    # 使用工具类进行批量处理
    utils.process_graphs_batch(
        algorithm_func=tabu_algorithm_handler,
        algorithm_name="Tabu"
    )