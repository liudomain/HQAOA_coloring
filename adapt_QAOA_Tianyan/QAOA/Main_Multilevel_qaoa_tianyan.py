from concurrent.futures import ProcessPoolExecutor, as_completed

import os
import sys
import math
import numpy as np  # 用于数值计算
# import mindspore as ms
import argparse
import traceback
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互模式，自动关闭图片
# 从 multilevel_common_tianyan.py 导入共享函数
from multilevel_common_tianyan import (
    divide_graph,
    smart_divide_graph_with_qubit_constraint,  # 智能子图划分
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
from multilevel_standard_QAOA_tianyan import (
    sequential_process_subgraphs_tianyan,  # 顺序处理子图着色（天衍平台版本）
    iterative_optimization_tianyan,  # 迭代优化着色方案（天衍平台版本）
    plot_energy_convergence,  # 能量收敛可视化
)

# 添加标准算法模块到路径，导入图加载模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from graph_loader import load_graphs_from_dir

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

def generate_simple_graph(num_nodes, num_edges, seed=10):
    """生成简单测试图"""
    np.random.seed(seed)
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    edges = []
    while len(edges) < num_edges:
        u, v = np.random.randint(0, num_nodes, 2)
        if u != v and (u, v) not in edges and (v, u) not in edges:
            edges.append((u, v))
            graph.add_edge(u, v)
    graph.file_name = f"simple_graph_{num_nodes}nodes_{len(edges)}edges_seed{seed}.col"
    return graph


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='天衍平台标准 QAOA 图着色算法'
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
    parser.add_argument('--train-max-iter', type=int, default=20,
                        help='参数训练最大迭代次数 (default: 20)')
    parser.add_argument('--train-lr', type=float, default=0.05,
                        help='参数训练学习率 (default: 0.05)')
    parser.add_argument('--train-epsilon-fd', type=float, default=0.05,
                        help='有限差分步长 (default: 0.05)')
    parser.add_argument('--enable-cache', action='store_true',
                        help='启用参数缓存 (默认: False)')
    parser.add_argument('--dataset', type=str, default='test_dataset',
                        help='数据集名称 (default: test_dataset)')
    return parser.parse_args()


# ---------- 标准 QAOA 独立入口（天衍平台版本）----------
def main_standard(graphs, dataset, graph_index, seed, platform, lab_id, trained_params=None,
                   train_params: bool = True, train_max_iter: int = 50, train_lr: float = 0.01):
    """
    标准 QAOA 入口（天衍平台版本）

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

    返回格式与 main() 完全一致：list[dict]
    """
    # ---- 复用 adapt 的目录、日志、参数配置 ----
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    CSV_DIR = os.path.join(BASE_DIR, "csvs")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "graph_visualizations"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "subgraph_visualizations"), exist_ok=True)

    subgraph_csv = os.path.join(CSV_DIR, "standard_subgraph_results.csv")
    if not os.path.exists(subgraph_csv):
        with open(subgraph_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "subgraph_index",
                "nodes", "edges", "min_k", "conflicts", "status", "processing_time"
            ])

    graph_log_csv = os.path.join(LOGS_DIR, "standard_graph_results.log")
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
    log_file = os.path.join(LOGS_DIR, "standard_qaoa.log")
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
            print(f"Processing Graph {index}/{len(graphs)}: {base_title} (Standard-QAOA Tianyan)")
            print(f"Graph Properties: {num_nodes} Nodes, {num_edges} Edges")
            print(f"{'='*50}")

            # 1. Original graph visualization (reuse from adapt function)
            try:
                filename = f"{base_title}_original"
                plot_original_graph(
                    graph,
                    title=f"{base_title} - Original Graph (Nodes: {num_nodes}, Edges: {num_edges})",
                    filename=filename,
                    output_dir=os.path.join(BASE_DIR, "graph_visualizations")
                )
            except Exception as e:
                handle_exception("plot_original_graph", index, e)

            # 2. 检测图的稀疏程度并优化子图划分参数
            # 天衍平台量子比特数上限: 36
            MAX_QUBITS = 36
            max_test_k = algorithm_params["max_k"]
            max_qubits_per_node = math.ceil(math.log2(max_test_k))
            max_nodes_by_qubits = MAX_QUBITS // max_qubits_per_node
            default_max_nodes = min(10, max_nodes_by_qubits)

            # 检测稀疏程度
            strategy, description, edge_density = detect_sparse_graph_strategy(graph)
            logger.info(f"图稀疏度检测: {description} (边密度: {edge_density:.4f})")

            # 计算最优的子图划分参数
            default_num_subgraphs = max(2, int(np.sqrt(num_nodes)))
            num_subgraphs, max_nodes, strategy_info = calculate_optimal_subgraph_params(
                graph, MAX_QUBITS, max_test_k, default_num_subgraphs, default_max_nodes
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
                max_qubits=MAX_QUBITS,  # 天衍平台最大量子比特数
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
                    output_dir=os.path.join(BASE_DIR, "subgraph_visualizations")
                )
                plot_Original_IDs_subgraphs(
                    subgraphs,
                    title=f"{base_title} - Subgraphs (Original IDs)",
                    filename=f"{base_title}_subgraphs_original",
                    output_dir=os.path.join(BASE_DIR, "subgraph_visualizations")
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

            print(f"\n===== Optimization Summary (Standard-QAOA) =====")
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
                    output_dir=os.path.join(BASE_DIR, "subgraph_visualizations")
                )
                plot_Original_IDs_colored_subgraphs(
                    subgraphs, subgraph_colorings,
                    title=f"{base_title} - Colored Subgraphs (Original IDs - Original Coloring)",
                    min_k_list=min_k_list, filename=f"{base_title}_colored_subgraphs_original",
                    output_dir=os.path.join(BASE_DIR, "subgraph_visualizations")
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
                    output_dir=os.path.join(BASE_DIR, "graph_visualizations"),
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
    print(f"Standard-QAOA all graphs processed, total time: {total_time:.1f}s")
    print(f"Successfully processed {len(all_results)}/{len(graphs)} graphs")
    print(f"CSV: {subgraph_csv} | Logs: {graph_log_csv}")
    print(f"{'='*50}")
    return all_results


# # ============ 使用示例 ============
# # Main_Multilevel_qaoa.py 使用示例
# python Main_Multilevel_qaoa.py --adapt --format-type col
# python Main_Multilevel_qaoa.py --standard --format-type pkl


# ============================================================================
# 主函数入口（天衍平台版本）
# ============================================================================

def main():
    """
    主函数：天衍平台标准 QAOA 着色
    """
    # 解析命令行参数
    args = parse_args()

    # ========== 配置参数 ==========
    # 从配置文件读取天衍平台密钥，如果没有配置则使用环境变量
    if TIANYAN_CONFIG and TIANYAN_CONFIG.get("login_key"):
        LOGIN_KEY = TIANYAN_CONFIG["login_key"]
    else:
        # 从环境变量读取，避免硬编码
        LOGIN_KEY = os.environ.get("TIANYAN_LOGIN_KEY", None)
        if LOGIN_KEY is None:
            raise ValueError("请在 config.py 中配置 TIANYAN_CONFIG['login_key'] 或设置环境变量 TIANYAN_LOGIN_KEY")

    LAB_ID = TIANYAN_CONFIG.get("lab_id") if TIANYAN_CONFIG else None  # 如果不设置，将自动创建实验室
    SEED = args.seed
    TRAIN_PARAMS = args.train_params  # 是否训练参数
    TRAIN_MAX_ITER = args.train_max_iter
    TRAIN_LR = args.train_lr
    DATASET = args.dataset
    GRAPH_DIR = args.graph_dir
    FORMAT_TYPE = args.format_type
    # =================================

    print("=" * 60)
    print("天衍平台标准 QAOA 图着色")
    print("=" * 60)

    # 加载图数据 - 根据 format_type 参数选择加载方式
    print(f"\n加载图数据...")
    print(f"  格式类型: {FORMAT_TYPE}")

    graphs = []

    if GRAPH_DIR:
        # 指定了目录，按指定格式加载
        print(f"  使用指定目录: {GRAPH_DIR}")
        graphs = load_graphs_from_dir(GRAPH_DIR, format_type=FORMAT_TYPE)
    else:
        # 没有指定目录，使用 .graph_loader 的默认目录查找
        print(f"  使用默认数据目录（优先 .col，其次 .pkl）")
        graphs = load_graphs_from_dir('default', format_type=FORMAT_TYPE)

    # 仍然没有数据，生成简单测试图
    if not graphs:
        print(f"  未找到图数据文件，生成简单测试图: 10 节点, 15 边 (seed: {SEED})")
        graph = generate_simple_graph(10, 15, SEED)
        graphs = [graph]

    if not graphs:
        print("⚠️ 未能加载任何图数据，程序退出")
        return

    # 打印图数据信息
    print(f"\n✓ 成功加载 {len(graphs)} 张图")
    for i, g in enumerate(graphs, 1):
        g_name = getattr(g, 'file_name', f'graph_{i}')
        print(f"  图 {i}: {g_name} - {g.number_of_nodes()} 节点, {g.number_of_edges()} 边")

    # 初始化天衍平台
    try:
        platform = TianYanPlatform(login_key=LOGIN_KEY)
        platform.set_machine("tianyan_sw")
        print("✅ 天衍平台初始化成功")
    except Exception as e:
        print(f"⚠️ 天衍平台初始化失败: {e}")
        return

    # 如果没有提供 lab_id，则创建实验室
    if LAB_ID is None:
        try:
            from datetime import datetime
            lab_name = f'multilevel_qaoa_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            lab_id = platform.create_lab(
                name=lab_name,
                remark='Multilevel QAOA Graph Coloring'
            )
            print(f"✅ 实验室创建成功: {lab_id}")
        except Exception as e:
            print(f"⚠️ 实验室创建失败: {e}")
            return
    else:
        lab_id = LAB_ID

    # 执行标准 QAOA
    print("\n" + "=" * 60)
    print("开始标准 QAOA 处理...")
    print("=" * 60)

    try:
        results = main_standard(
            graphs=graphs,
            dataset=DATASET,
            graph_index=0,
            seed=SEED,
            platform=platform,
            lab_id=lab_id,
            trained_params=None,
            train_params=TRAIN_PARAMS,
            train_max_iter=TRAIN_MAX_ITER,
            train_lr=TRAIN_LR
        )

        # 输出结果
        print("\n" + "=" * 60)
        print("测试结果")
        print("=" * 60)
        for r in results:
            print(f"Graph {r['graph_index']}: Colors={r['unique_colors']}, "
                  f"Conflicts={r['final_conflicts']}, "
                  f"Time={r['processing_time']:.2f}s, "
                  f"Accuracy={r['accuracy']:.4f}")
        print("=" * 60)

    except Exception as e:
        print(f"\n⚠️ 处理失败: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
'''
## 加载 pkl 数据
python Main_Multilevel_qaoa_tianyan.py --format-type pkl

# 加载 col 数据
python Main_Multilevel_qaoa_tianyan.py --format-type col

# 自动选择（优先 pkl，然后 col）
python Main_Multilevel_qaoa_tianyan.py --format-type auto

# 指定自定义目录
python Main_Multilevel_qaoa_tianyan.py --graph-dir "d:\path\to\pkl\files" --format-type pkl

# 完整参数示例
python Main_Multilevel_qaoa_tianyan.py --format-type pkl --train-params --train-max-iter 10 --train-lr 0.01
'''
