from concurrent.futures import ProcessPoolExecutor, as_completed

import os
import sys
import math
import numpy as np  # 用于数值计算
# import mindspore as ms
import argparse
import traceback
# 从 multilevel_common_tianyan.py 导入共享函数
from multilevel_common_tianyan import (
    divide_graph,
    smart_divide_graph_with_qubit_constraint,
    detect_sparse_graph_strategy,
    calculate_optimal_subgraph_params,
    count_conflicts,
    plot_original_graph,
    plot_New_IDs_subgraphs,
    plot_Original_IDs_subgraphs,
    plot_New_IDs_colored_subgraphs,
    plot_Original_IDs_colored_subgraphs,
    get_subgraph_coloring,
    visualize_graph,
    handle_exception,
)
# 从天衍平台版本导入 QAOA 函数
from multilevel_standard_QAOA_tianyan_large import (
    sequential_process_subgraphs_tianyan,  # 顺序处理子图着色（天衍平台版本）
    iterative_optimization_tianyan,  # 迭代优化着色方案（天衍平台版本）
    plot_energy_convergence,  # 能量收敛可视化
)

# 添加标准算法模块到路径，导入图加载模块
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_FILE_DIR)
sys.path.insert(0, PARENT_DIR)
from graph_loader import read_col_file

# 导入配置文件
try:
    from config import TIANYAN_CONFIG
except ImportError:
    TIANYAN_CONFIG = None



import csv, time, traceback
import json
import logging
from cqlib import TianYanPlatform

#执行相关算法逻辑，并存储结果为csv/log文件，便于后续分析

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='天衍平台标准 QAOA 图着色算法（朴素划分版本）'
    )
    parser.add_argument('--graph-dir', type=str, default=None,
                        help='图数据目录路径（指定后从此目录加载图文件）')
    parser.add_argument('--format-type', type=str, default='auto',
                        choices=['auto', 'col', 'pkl'],
                        help='数据加载格式: auto(自动), col(.col only), pkl(.pkl only) (default: auto)')
    parser.add_argument('--seed', type=int, default=10,
                        help='随机种子 (default: 10)')
    parser.add_argument('--train-params', action='store_true',
                        help='是否训练参数 (默认: False)')
    parser.add_argument('--train-max-iter', type=int, default=100,
                        help='参数训练最大迭代次数 (default: 100)')
    parser.add_argument('--train-lr', type=float, default=0.01,
                        help='参数训练学习率 (default: 0.01)')
    parser.add_argument('--dataset', type=str, default='test_dataset',
                        help='数据集名称 (default: test_dataset)')
    parser.add_argument('--max-qubits', type=int, default=200,
                        help='量子比特数限制（根据机时包限制调整，默认: 200）')
    return parser.parse_args()


# ---------- 标准 QAOA 独立入口（天衍平台版本，朴素划分）----------
def main_standard(graphs, dataset, graph_index, seed, platform, lab_id, trained_params=None,
                   train_params: bool = True, train_max_iter: int = 50, train_lr: float = 0.01, max_qubits: int = 200):
    """
    标准 QAOA 入口（天衍平台版本，朴素划分）

    参数:
        graphs: 图列表
        dataset: 数据集名称
        graph_index: 图索引
        seed: 随机种子
        platform: TianYanPlatform 实例
        lab_id: 实验室ID
        trained_params: 预训练参数字典
        train_params: 是否训练参数（默认 True）
        train_max_iter: 参数训练最大迭代次数
        train_lr: 参数训练学习率
        max_qubits: 量子比特数限制（根据机时包限制调整）

    返回格式与 main() 完全一致：list[dict]
    """
    # ---- 复用 adapt 的目录、日志、参数配置 ----
    # 当前文件在: HAdaQAOA/HadaQAOA/adapt_QAOA_Tianyan/QAOA/
    # 需要回到 adapt_QAOA_Tianyan/ 目录
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    CSV_DIR = os.path.join(BASE_DIR, "csvs")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "large_graph_visualizations"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "large_subgraph_visualizations"), exist_ok=True)

    subgraph_csv = os.path.join(CSV_DIR, "large_standard_naive_subgraph_results.csv")
    if not os.path.exists(subgraph_csv):
        with open(subgraph_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "subgraph_index",
                "nodes", "edges", "min_k", "conflicts", "status", "processing_time"
            ])

    graph_log_csv = os.path.join(LOGS_DIR, "large_standard_naive_graph_results.log")
    if not os.path.exists(graph_log_csv):
        with open(graph_log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "nodes", "edges",
                "final_conflicts", "total_edges", "final_accuracy",
                "unique_colors", "global_max_k", "best_k_value",
                "subgraph_reoptimization_count", "processing_time",
                "conflict_changes", "total_time", "train_params",
                "train_max_iter", "train_lr"
            ])

    # 配置日志
    log_file = os.path.join(LOGS_DIR, "large_standard_naive_qaoa.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # 抑制天衍平台 SDK 的等待日志
    tianyan_logger = logging.getLogger('cqlib')
    tianyan_logger.setLevel(logging.WARNING)

    all_results = []
    total_start_time = time.time()
    algorithm_params = {
        "max_k": 20,
        "p": 1,
        "num_shots": 1000,
        "max_iter": 20,
        "early_stop_threshold": 5,
        "Q": 20
    }

    # ---------- 开始逐图处理 ----------
    for index, graph in enumerate(graphs, start=1):
        graph_start_time = time.time()
        try:
            graph_name = getattr(graph, "file_name", f"graph_{index}")
            base_title = os.path.splitext(graph_name)[0]
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()

            print(f"\n{'='*50}")
            print(f"Processing Graph {index}/{len(graphs)}: {base_title} (Standard-QAOA Tianyan Large Naive)")
            print(f"Graph Properties: {num_nodes} Nodes, {num_edges} Edges")
            print(f"{'='*50}")

            # 1. Original graph visualization (reuse from adapt function)
            try:
                filename = f"{base_title}_original"
                plot_original_graph(
                    graph,
                    title=f"{base_title} - Original Graph (Nodes: {num_nodes}, Edges: {num_edges})",
                    filename=filename,
                    output_dir=os.path.join(BASE_DIR, "large_graph_visualizations")
                )
            except Exception as e:
                handle_exception("plot_original_graph", index, e)

            # 2. 检测图的稀疏程度并优化子图划分参数
            max_test_k = algorithm_params["max_k"]
            max_qubits_per_node = math.ceil(math.log2(max_test_k))
            max_nodes_by_qubits = max_qubits // max_qubits_per_node
            default_max_nodes = min(10, max_nodes_by_qubits)

            # 检测稀疏程度
            strategy, description, edge_density = detect_sparse_graph_strategy(graph)
            logger.info(f"图稀疏度检测: {description} (边密度: {edge_density:.4f})")

            # 计算最优的子图划分参数
            default_num_subgraphs = max(2, int(np.sqrt(num_nodes)))
            num_subgraphs, max_nodes, strategy_info = calculate_optimal_subgraph_params(
                graph, max_qubits, max_test_k, default_num_subgraphs, default_max_nodes
            )
            logger.info(f"子图划分策略: {strategy_info}")
            logger.info(f"调整后参数: 子图数量={num_subgraphs}, 最大节点数={max_nodes}")

            # 3. 使用朴素子图划分（朴素划分版本，不退化为贪心着色）
            # divide_graph 使用固定的子图数量和节点限制进行划分
            subgraphs, sub_mappings, divide_info = divide_graph(
                graph,
                num_subgraphs=num_subgraphs,  # 使用计算得到的子图数量
                max_nodes=max_nodes,  # 使用计算得到的最大节点数
                Q=algorithm_params["Q"],
                max_qubits=max_qubits,  # 天衍平台最大量子比特数
                max_k=algorithm_params["max_k"]
            )
            logger.info(f"朴素划分完成: {len(subgraphs)} 个子图")
            logger.info(f"  最大节点数/子图: {max_nodes if isinstance(divide_info, dict) else 'N/A'}")

            # 4. Subgraph visualization (optional)
            try:
                plot_New_IDs_subgraphs(
                    subgraphs, sub_mappings,
                    title=f"{base_title} - Subgraphs (Renumbered)",
                    filename=f"{base_title}_subgraphs_renumbered",
                    output_dir=os.path.join(BASE_DIR, "large_subgraph_visualizations")
                )
                plot_Original_IDs_subgraphs(
                    subgraphs,
                    title=f"{base_title} - Subgraphs (Original IDs)",
                    filename=f"{base_title}_subgraphs_original",
                    output_dir=os.path.join(BASE_DIR, "large_subgraph_visualizations")
                )
            except Exception as e:
                handle_exception("subgraph plotting", index, e)

            # 5. 标准 QAOA 子图处理（天衍平台版本）
            subgraph_start_time = time.time()
            subgraph_results = sequential_process_subgraphs_tianyan(
                subgraphs=subgraphs,
                sub_mappings=sub_mappings,
                dataset_name=dataset,
                graph_id=index,
                platform=platform,
                lab_id=lab_id,
                max_k=algorithm_params["max_k"],
                p=algorithm_params["p"],
                num_shots=algorithm_params["num_shots"],
                train_params=train_params,
                train_max_iter=train_max_iter,
                train_lr=train_lr,
                algorithm='standard',
                graph_name=graph_name
            )
            subgraph_total_time = time.time() - subgraph_start_time

            # 记录子图级别结果到 CSV
            dataset_name = os.path.basename(getattr(graph, "file_name", "unknown").split(os.sep)[0])
            for sub_idx, result in enumerate(subgraph_results):
                if result is None:
                    continue
                min_k, coloring, conflicts, status, _ = result
                subgraph = subgraphs[sub_idx] if sub_idx < len(subgraphs) else None
                sub_nodes = subgraph.number_of_nodes() if subgraph else 0
                sub_edges = subgraph.number_of_edges() if subgraph else 0
                with open(subgraph_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        dataset_name, graph_name, index, sub_idx + 1,
                        sub_nodes, sub_edges, min_k, conflicts, status,
                        round(subgraph_total_time / len(subgraphs), 4) if subgraphs else 0
                    ])

            # 6. 标准 QAOA 迭代优化（天衍平台版本）
            optimized_coloring, opt_acc, conflict_counts, conflict_history, training_info = iterative_optimization_tianyan(
                graph=graph,
                subgraphs=subgraphs,
                sub_mappings=sub_mappings,
                subgraph_results=subgraph_results,
                max_k=algorithm_params["max_k"],
                p=algorithm_params["p"],
                max_iter=algorithm_params["max_iter"],
                early_stop_threshold=algorithm_params["early_stop_threshold"],
                dataset_name=dataset,
                graph_id=index
            )
            subgraph_opt_history = []  # compatibility placeholder

            # 7. 统一输出逻辑（与 adapt 完全一致）
            final_coloring = optimized_coloring
            unique_colors = len(set(final_coloring.values())) if final_coloring else 0
            final_conflicts = count_conflicts(final_coloring, graph) if final_coloring else -1
            reoptimization_count = 0  # 兼容性
            # 确保 min_k_list 与 subgraphs 一一对应，保留 None 值
            min_k_list = [r[0] if r is not None else None for r in subgraph_results]
            best_k_value = min(unique_colors, max([k for k in min_k_list if k is not None]) if min_k_list else unique_colors)

            print(f"\n===== Optimization Summary (Standard-QAOA Large Naive) =====")
            print(f"Final Conflicts: {final_conflicts} (Total Edges: {num_edges})")
            print(f"Final Accuracy: {opt_acc:.4f}")
            print(f"Colors Used: {unique_colors} (Global max_k limit: {algorithm_params['max_k']})")
            print(f"Best k Value: {best_k_value}")
            print(f"Subgraph Reoptimization Count: {reoptimization_count}")

            # 8. 子图着色可视化（使用子图原始着色结果，而非全局优化结果）
            try:
                # 从 subgraph_results 中提取原始着色结果
                subgraph_colorings = []
                for i, result in enumerate(subgraph_results):
                    if result is not None:
                        _, coloring, conflicts, status, _ = result
                        if isinstance(coloring, dict) and coloring:
                            subgraph_colorings.append(coloring)
                        else:
                            # 空着色，使用get_subgraph_coloring兜底
                            mk = min_k_list[i] if i < len(min_k_list) else 2
                            subgraph = subgraphs[i]
                            sub_coloring = get_subgraph_coloring(subgraph, final_coloring, mk)
                            subgraph_colorings.append(sub_coloring)
                    else:
                        # 无结果，使用get_subgraph_coloring兜底
                        mk = min_k_list[i] if i < len(min_k_list) else 2
                        subgraph = subgraphs[i]
                        sub_coloring = get_subgraph_coloring(subgraph, final_coloring, mk)
                        subgraph_colorings.append(sub_coloring)

                plot_New_IDs_colored_subgraphs(
                    subgraphs, subgraph_colorings, sub_mappings, min_k_list,
                    title=f"{base_title} - Colored Subgraphs (Renumbered - Original Coloring)",
                    filename=f"{base_title}_colored_subgraphs_renumbered",
                    output_dir=os.path.join(BASE_DIR, "large_subgraph_visualizations")
                )
                plot_Original_IDs_colored_subgraphs(
                    subgraphs, subgraph_colorings,
                    title=f"{base_title} - Colored Subgraphs (Original IDs - Original Coloring)",
                    min_k_list=min_k_list, filename=f"{base_title}_colored_subgraphs_original",
                    output_dir=os.path.join(BASE_DIR, "large_subgraph_visualizations")
                )
            except Exception as e:
                print(f"Error in colored subgraph plotting for graph {index}: {str(e)}")
                traceback.print_exc()

            # 9. 计算当前图的处理时间
            graph_time = time.time() - graph_start_time

            # 10. Final graph visualization
            try:
                final_graph_title = (
                    f"{base_title}\n"
                    f"Coloring Result (Colors: {unique_colors}, "
                    f"Nodes: {num_nodes}, Edges: {num_edges}, "
                    f"Conflicts: {final_conflicts})"
                )
                visualize_graph(
                    graph, coloring=final_coloring, title=final_graph_title,
                    index=index, min_k=unique_colors,
                    filename=f"{base_title}_final_coloring",
                    output_dir=os.path.join(BASE_DIR, "large_graph_visualizations"),
                    processing_time=graph_time
                )
            except Exception as e:
                handle_exception("visualize_graph", index, e)

            # 11. 收集结果
            result = {
                "graph_index": index,
                "graph": graph,
                "final_coloring": final_coloring,
                "subgraphs": subgraphs,
                "sub_mappings": sub_mappings,
                "subgraph_results": subgraph_results,
                "sub_colorings": subgraph_colorings,
                "conflict_counts": conflict_counts,
                "conflict_history": conflict_history,
                "subgraph_opt_history": subgraph_opt_history,
                "unique_colors": unique_colors,
                "final_conflicts": final_conflicts,
                "accuracy": opt_acc,
                "processing_time": graph_time,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "base_title": base_title,
                "global_max_k": algorithm_params["max_k"],
                "best_k_value": best_k_value,
                "reoptimization_count": reoptimization_count
            }
            all_results.append(result)

            # 12. 写全局日志
            with open(graph_log_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                conflict_changes_str = ",".join(map(str, conflict_counts)) if conflict_counts else "N/A"
                writer.writerow([
                    dataset_name, graph_name, index,
                    num_nodes, num_edges, final_conflicts, num_edges,
                    round(opt_acc, 4), unique_colors,
                    algorithm_params["max_k"], best_k_value,
                    reoptimization_count, round(graph_time, 4),
                    conflict_changes_str,
                    round(time.time() - total_start_time, 4),
                    train_params, train_max_iter, train_lr
                ])

        except Exception as e:
            print(f"Uncaught exception while processing graph {index}: {e}")
            traceback.print_exc()
            continue

    # ---------- Post-processing ----------
    # 13. 计算总时间
    total_time = time.time() - total_start_time
    print(f"\n{'='*50}")
    print(f"Standard-QAOA Large Naive all graphs processed, total time: {total_time:.1f}s")
    print(f"Successfully processed {len(all_results)}/{len(graphs)} graphs")
    print(f"CSV: {subgraph_csv} | Logs: {graph_log_csv}")
    print(f"{'='*50}")
    return all_results


# ============================================================================
# 说明：本文件已被简化，主函数已移除
# ============================================================================
#
# 本文件包含朴素划分版本的标准 QAOA 实现（大规模图），但主函数入口已移至 Main_Multilevel_qaoa_tianyan.py
# 如需使用朴素划分功能，请调用本文件中的 main_standard() 函数
#
# 使用说明：
# # 指定量子比特数限制（根据机时包限制调整）
# python Main_Multilevel_qaoa_tianyan_large_naive.py --max-qubits 200
# python Main_Multilevel_qaoa_tianyan_large_naive.py --max-qubits 32  # 更小的限制
#
# # 完整参数示例
# python Main_Multilevel_qaoa_tianyan_large_naive.py --train-params --train-max-iter 100 --train-lr 0.01 --max-qubits 200
#
# 注意：
# |- 如果出现 "最大比特数不支持本任务" 错误，请使用 --max-qubits 参数调整
# |- 默认值是 200，可以根据您的机时包限制降低此值
# |- 较小的 max_qubits 值会导致更小的子图，可能影响着色质量
#
# ============================================================================
'''
- 如果出现 "最大比特数不支持本任务" 错误，请使用 --max-qubits 参数调整
- 默认值是 200，可以根据您的机时包限制降低此值
- 较小的 max_qubits 值会导致更小的子图，可能影响着色质量
'''
