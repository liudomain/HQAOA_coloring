from concurrent.futures import ProcessPoolExecutor, as_completed

import os
import sys
import math
import numpy as np  # 用于数值计算
import mindspore as ms
import argparse
import traceback
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互模式，自动关闭图片
# 从 multilevel_common.py 导入共享函数
from multilevel_common import (
    divide_graph,
    smart_divide_graph_with_qubit_constraint,  # 智能子图划分，支持量子比特约束
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
# 从三个专门模块导入各自特有的函数
from multilevel_adapt_QAOA_k_coloring import (
    sequential_process_subgraphs,  # 顺序处理子图着色
    iterative_optimization,  # 迭代优化着色方案
)
from multilevel_standard_QAOA_k_coloring import (
    sequential_process_subgraphs_standard,  # 顺序处理子图着色
    iterative_optimization_standard,  # 迭代优化着色方案
)
from multilevel_adapt_noise_QAOA_k_coloring import (
    sequential_process_subgraphs_noise,  # 顺序处理子图着色
    iterative_optimization_noise,  # 迭代优化着色方案
)
from graph_loader import load_graphs_from_dir, read_col_file
import csv, os, time, traceback, json, logging

# 添加经典算法模块路径
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_FILE_DIR)  # HadaQAOA
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, "classical_algorithms"))

from greedy import GreedyColoring
from Backtracking_and_Welch_Powell import process_graph_with_welch_powell


def run_greedy_coloring(graph, filename):
    """运行贪心算法进行图着色"""
    import time
    start_time = time.perf_counter()
    try:
        greedy = GreedyColoring(graph)
        coloring, num_colors = greedy.execute()
        conflicts = count_conflicts(coloring, graph)
        exec_time = (time.perf_counter() - start_time) * 1000  # 毫秒
        return {
            'algorithm': 'Greedy',
            'filename': filename,
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'num_colors': num_colors,
            'conflicts': conflicts,
            'execution_time_ms': round(exec_time, 4),
            'is_valid': conflicts == 0,
            'coloring': coloring
        }
    except Exception as e:
        print(f"❌ 贪心算法执行失败: {e}")
        traceback.print_exc()
        return None


def run_welch_powell_coloring(graph, filename):
    """运行Welch-Powell算法进行图着色"""
    try:
        result = process_graph_with_welch_powell(graph, filename)
        if result:
            result['algorithm'] = 'WelchPowell'
        return result
    except Exception as e:
        print(f"❌ Welch-Powell算法执行失败: {e}")
        traceback.print_exc()
        return None


def validate_graph_feasibility(graph, max_nodes_per_subgraph):
    """验证图的可处理性"""
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    edge_density = 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

    validation_result = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "edge_density": edge_density,
        "is_feasible": num_nodes <= max_nodes_per_subgraph,
        "recommended_subgraphs": max(2, math.ceil(num_nodes / max_nodes_per_subgraph))
    }
    return validation_result

#执行相关算法逻辑，并存储结果为csv/log文件，便于后续分析
def main_adapt(graphs, dataset, graph_index, seed):
    """Adapt-QAOA 主入口：完整流程与输出，带统一标识便于区分"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    CSV_DIR = os.path.join(BASE_DIR, "csvs")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    # os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "graph_visualizations"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "subgraph_visualizations"), exist_ok=True)

    subgraph_csv   = os.path.join(CSV_DIR, "adapt_subgraph_results.csv")
    graph_log_csv  = os.path.join(LOGS_DIR, "adapt_graph_results.log")
    if not os.path.exists(subgraph_csv):
        with open(subgraph_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "subgraph_index",
                "nodes", "edges", "min_k", "conflicts", "status", "processing_time"
            ])
    if not os.path.exists(graph_log_csv):
        with open(graph_log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "nodes", "edges",
                "final_conflicts", "total_edges", "final_accuracy",
                "unique_colors", "global_max_k", "best_k_value",
                "subgraph_reoptimization_count", "processing_time",
                "conflict_changes", "total_time"
            ])

    all_results = []
    total_start = time.time()
    algo_params = {
        "n_qubits_per_node": 2,
        "learning_rate": 0.01,
        "max_k": 20,
        "p": 3,
        "num_steps": 1000,
        "max_iter": 10,
        "adjacency_threshold": 0.3,
        "early_stop_threshold": 3,
        "penalty": 1000,
        "Q": 20
    }

    for idx, graph in enumerate(graphs, start=1):
        np.random.seed(seed + idx)
        g_start = time.time()
        try:
            g_name   = getattr(graph, "file_name", f"graph_{idx}")
            base_ttl = os.path.splitext(g_name)[0]
            n_nodes  = graph.number_of_nodes()
            n_edges  = graph.number_of_edges()

            print(f"\n{'='*50}")
            # print(f"Adapt-QAOA | 图 {idx}/{len(graphs)}: {base_ttl}")
            print(f"adapt_qaoa | 图 {idx}/{len(graphs)}: {base_ttl}")
            print(f"节点: {n_nodes} | 边: {n_edges}")
            print(f"{'='*50}")

            # 1 原始图
            try:
                plot_original_graph(
                    graph,
                    title=f"[Adapt-QAOA] {base_ttl} - Original Graph (Nodes: {n_nodes}, Edges: {n_edges})"
                )
            except Exception as e:
                handle_exception("plot_original_graph", idx, e)

            # 2 智能子图划分（限制量子比特数最多为21）
            max_qubits = 21
            subgraphs, sub_maps, divide_info = smart_divide_graph_with_qubit_constraint(
                graph,
                max_qubits=max_qubits,
                max_k_per_subgraph=algo_params["max_k"],
                Q=algo_params["Q"]
            )
            print(f"Adapt-QAOA 智能划分完成：{len(subgraphs)} 个子图（量子比特约束：≤{max_qubits}）")

            # 3 子图可视化
            try:
                plot_New_IDs_subgraphs(
                    subgraphs, sub_maps,
                    title=f"[Adapt-QAOA] {base_ttl} - Subgraphs (Renumbered)"
                )
                plot_Original_IDs_subgraphs(
                    subgraphs,
                    title=f"[Adapt-QAOA] {base_ttl} - Subgraphs (Original IDs)"
                )
            except Exception as e:
                handle_exception("subgraph plotting", idx, e)

            # 4 子图着色
            sub_start = time.time()
            sub_results = sequential_process_subgraphs(
                subgraphs=subgraphs,
                sub_mappings=sub_maps,
                dataset_name=dataset,
                graph_id=idx,
                max_k=algo_params["max_k"],
                p=algo_params["p"],
                num_steps=algo_params["num_steps"],
                vertex_colors=None,
                nodes_to_recolor=None,
                penalty=algo_params["penalty"],
                Q=algo_params["Q"],
                learning_rate=algo_params["learning_rate"]
            )
            sub_time = time.time() - sub_start
            min_k_list = [r[0] for r in sub_results if r and r[0] is not None]

            # 写子图 CSV
            dataset_name = os.path.basename(getattr(graph, "file_name", "unknown").split(os.sep)[0])
            for s_idx, res in enumerate(sub_results):
                if res is None:
                    continue
                mk, _, conf, stat, _ = res
                sg = subgraphs[s_idx] if s_idx < len(subgraphs) else None
                with open(subgraph_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        dataset_name, g_name, idx, s_idx + 1,
                        sg.number_of_nodes() if sg else 0,
                        sg.number_of_edges() if sg else 0,
                        mk, conf, stat,
                        round(sub_time / len(subgraphs), 4) if subgraphs else 0
                    ])

            # 5 迭代优化
            opt_color, opt_acc, conf_counts, conf_hist, sub_opt_hist = iterative_optimization(
                graph=graph,
                subgraphs=subgraphs,
                sub_mappings=sub_maps,
                subgraph_results=sub_results,
                max_k=algo_params["max_k"],
                p=algo_params["p"],
                num_steps=algo_params["num_steps"],
                max_iter=algo_params["max_iter"],
                adjacency_threshold=algo_params["adjacency_threshold"],
                early_stop_threshold=algo_params["early_stop_threshold"],
                penalty=algo_params["penalty"],
                Q=algo_params["Q"],
                learning_rate=algo_params["learning_rate"],
                vertex_colors=None,
                nodes_to_recolor=None,
                dataset_name=dataset,
                graph_id=idx
            )

            final_color = opt_color
            uniq_colors = len(set(final_color.values())) if final_color else 0
            final_conf  = count_conflicts(final_color, graph) if final_color else -1
            reopt_cnt   = sum(1 for h in sub_opt_hist if isinstance(h, tuple) and len(h) >= 4 and h[3] > 0)
            best_k      = min(uniq_colors, max(min_k_list) if min_k_list else uniq_colors)

            print(f"\n===== Adapt-QAOA Optimization Summary =====")
            print(f"Final Conflicts: {final_conf} (Total Edges: {n_edges})")
            print(f"Final Accuracy: {opt_acc:.4f}")
            print(f"Colors Used: {uniq_colors}")
            print(f"Best k Value: {best_k}")
            print(f"Subgraph Reoptimization Count: {reopt_cnt}")

            # 6 子图着色可视化
            try:
                sub_colorings = [
                    get_subgraph_coloring(sg, final_color, mk)
                    for sg, mk in zip(subgraphs, min_k_list)
                ]
                plot_New_IDs_colored_subgraphs(
                    subgraphs, sub_colorings, sub_maps, min_k_list,
                    title=f"[Adapt-QAOA] {base_ttl} - Subgraph Coloring (Renumbered)",
                    filename=f"adapt_qaoa_{base_ttl}",
                    output_dir=os.path.join(BASE_DIR, "subgraph_visualizations")
                )
                plot_Original_IDs_colored_subgraphs(
                    subgraphs, sub_colorings,
                    title=f"[Adapt-QAOA] {base_ttl} - Subgraph Coloring (Original IDs)",
                    min_k_list=min_k_list,
                    filename=f"adapt_qaoa_{base_ttl}",
                    output_dir=os.path.join(BASE_DIR, "subgraph_visualizations")
                )
            except Exception as e:
                print(f"Adapt-QAOA Subgraph coloring visualization failed: {e}")
                traceback.print_exc()

            # 7 计算当前图的处理时间
            g_time = time.time() - g_start

            # 8 最终图可视化
            try:
                vis_title = (
                    f"[Adapt-QAOA] {base_ttl}\n"
                    f"Coloring Result (Colors: {uniq_colors}, "
                    f"Nodes: {n_nodes}, Edges: {n_edges}, Conflicts: {final_conf})"
                )
                visualize_graph(
                    graph, coloring=final_color, title=vis_title,
                    index=idx, min_k=uniq_colors, filename=f"adapt_{base_ttl}",
                    processing_time=g_time
                )
            except Exception as e:
                handle_exception("visualize_graph", idx, e)

            # 10. 收集结果
            all_results.append({
                "graph_index": idx,
                "graph": graph,
                "final_coloring": final_color,
                "subgraphs": subgraphs,
                "sub_mappings": sub_maps,
                "subgraph_results": sub_results,
                "sub_colorings": sub_colorings,
                "conflict_counts": conf_counts,
                "conflict_history": conf_hist,
                "subgraph_opt_history": sub_opt_hist,
                "unique_colors": uniq_colors,
                "final_conflicts": final_conf,
                "accuracy": opt_acc,
                "processing_time": g_time,
                "num_nodes": n_nodes,
                "num_edges": n_edges,
                "base_title": base_ttl,
                "global_max_k": algo_params["max_k"],
                "best_k_value": best_k,
                "reoptimization_count": reopt_cnt
            })

            # 10 写全局日志
            with open(graph_log_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    dataset_name, g_name, idx,
                    n_nodes, n_edges, final_conf, n_edges,
                    round(opt_acc, 4), uniq_colors,
                    algo_params["max_k"], best_k,
                    reopt_cnt, round(g_time, 4),
                    ",".join(map(str, conf_counts)) if conf_counts else "N/A",
                    round(time.time() - total_start, 4)
                ])
            print(f"Adapt-QAOA Graph {idx} completed, time: {g_time:.1f}s")

        except Exception as e:
            print(f"Adapt-QAOA 处理图 {idx} 异常: {e}")
            traceback.print_exc()
            continue

    # 11 生成 PDF 汇总
    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"Adapt-QAOA All completed, total time: {total_time:.1f}s")
    print(f"Successfully processed {len(all_results)}/{len(graphs)} graphs")
    print(f"Logs: {graph_log_csv} | CSV: {subgraph_csv}")
    print(f"{'='*50}")

    return all_results


# ----------- 测试主函数 -----------
def parse_test_args():
    """Parse command line arguments for testing"""
    parser = argparse.ArgumentParser(
        description='QAOA Algorithm Testing Tool: Supports Adapt-QAOA, Standard-QAOA, and Noisy Adapt-QAOA'
    )
    parser.add_argument('--adapt', action='store_true', help='Test Adapt-QAOA algorithm')
    parser.add_argument('--standard', action='store_true', help='Test Standard-QAOA algorithm')
    parser.add_argument('--adapt-noise', action='store_true', help='Test Noisy Adapt-QAOA algorithm')
    parser.add_argument('--noise-prob', type=float, default=0.05,
                        help='Noise probability for noisy experiments (default: 0.05)')
    parser.add_argument('--seed', type=int, default=10, help='Random seed (default: 10)')
    parser.add_argument('--dataset', type=str, default='test_dataset', help='Dataset name')
    parser.add_argument('--graph-index', type=int, default=0, help='Graph index')
    parser.add_argument('--graph-dir', type=str, default=None,
                        help='Graph data directory path, load graph files from this directory if specified')
    parser.add_argument('--format-type', type=str, default='auto',
                        choices=['auto', 'col', 'pkl'],
                        help='Data loading format: auto(automatic), col(.col only), pkl(.pkl only) (default: auto)')
    parser.add_argument('--large-datasets', action='store_true',
                        help='Run on large-scale datasets (cora.col, citeseer.col, pubmed.col)')
    parser.add_argument('--run-classical', action='store_true',
                        help='Run classical algorithms (Greedy, Welch-Powell) for comparison')
    return parser.parse_args()


def main_test():
    """Main testing function: Execute selected QAOA algorithms"""
    args = parse_test_args()

    # Validate that at least one algorithm is selected
    if not any([args.adapt, args.standard, args.adapt_noise]):
        print("⚠️ Must select at least one algorithm (--adapt/--standard/--adapt-noise)")
        print("Usage examples:")
        print("  python Main_Multilevel_qaoa.py --adapt")
        print("  python Main_Multilevel_qaoa.py --standard")
        print("  python Main_Multilevel_qaoa.py --adapt-noise --noise-prob 0.1")
        print("  python Main_Multilevel_qaoa.py --adapt --standard --adapt-noise")
        return

    # Load test graphs
    print("\n" + "="*60)
    print("Loading test graph data...")
    print("="*60)

    if args.graph_dir:
        graph_dir = os.path.abspath(args.graph_dir)
        print(f"Loading graph data from directory: {graph_dir} (format: {args.format_type})")
        graphs = load_graphs_from_dir(graph_dir, format_type=args.format_type)
    else:
        # Use default directory (graph_loader will select based on format_type)
        print(f"Using default graph directory (format: {args.format_type})")
        graphs = load_graphs_from_dir('default', format_type=args.format_type)

    if not graphs:
        print("⚠️ No graph data loaded, exiting program")
        return {}

    print(f'📦 Total {len(graphs)} test graphs (seed {args.seed})')

    # Display test configuration
    print("\n" + "="*60)
    print("Test Configuration:")
    print("="*60)
    print(f"  - Adapt-QAOA: {'Enabled' if args.adapt else 'Disabled'}")
    print(f"  - Standard-QAOA: {'Enabled' if args.standard else 'Disabled'}")
    print(f"  - Noisy Adapt-QAOA: {'Enabled' if args.adapt_noise else 'Disabled'}")
    if args.adapt_noise:
        print(f"  - Noise Probability: {args.noise_prob}")
    print(f"  - Random Seed: {args.seed}")
    print("="*60)

    results = {}

    # Execute Adapt-QAOA
    if args.adapt:
        print("\n" + "="*60)
        print("Starting Adapt-QAOA test...")
        print("="*60)
        try:
            adapt_results = main_adapt(graphs, args.dataset, args.graph_index, args.seed)
            results['adapt'] = adapt_results
            print(f"\n✅ Adapt-QAOA test completed, processed {len(adapt_results)} graphs")
        except Exception as e:
            print(f"\n⚠️ Adapt-QAOA test failed: {e}")
            import traceback
            traceback.print_exc()

    # Execute Standard-QAOA
    if args.standard:
        print("\n" + "="*60)
        print("Starting Standard-QAOA test...")
        print("="*60)
        try:
            standard_results = main_standard(graphs, args.dataset, args.graph_index, args.seed)
            results['standard'] = standard_results
            print(f"\n✅ Standard-QAOA test completed, processed {len(standard_results)} graphs")
        except Exception as e:
            print(f"\n⚠️ Standard-QAOA test failed: {e}")
            import traceback
            traceback.print_exc()

    # Execute Noisy Adapt-QAOA
    if args.adapt_noise:
        print("\n" + "="*60)
        print("Starting Noisy Adapt-QAOA test...")
        print("="*60)
        try:
            noise_results = main_adapt_noise(
                graphs, args.dataset, args.graph_index, args.seed,
                depolarizing_prob=args.noise_prob
            )
            results['adapt_noise'] = noise_results
            print(f"\n✅ Noisy Adapt-QAOA test completed, processed {len(noise_results)} graphs")
        except Exception as e:
            print(f"\n⚠️ Noisy Adapt-QAOA test failed: {e}")
            import traceback
            traceback.print_exc()

    # Output test results summary
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    for algo_name, algo_results in results.items():
        if algo_results:
            print(f"\n{algo_name.upper()}:")
            for r in algo_results:
                print(f"  Graph {r['graph_index']}: Colors={r['unique_colors']}, "
                      f"Conflicts={r['final_conflicts']}, "
                      f"Time={r['processing_time']:.2f}s, "
                      f"Accuracy={r['accuracy']:.4f}")
    print("\n" + "="*60)
    print("✅ All tests completed")
    print("="*60)


    return results


# ---------- 标准 QAOA 独立入口 ----------
def main_standard(graphs, dataset, graph_index, seed):
    """
    标准 QAOA 入口，流程与 adapt 完全一致，仅把子图处理替换成 standard 系列函数
    返回格式与 main() 完全一致：list[dict]
    """
    # ---- 复用 adapt 的目录、日志、参数配置 ----
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    CSV_DIR = os.path.join(BASE_DIR, "csvs")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    # os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
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
                "conflict_changes", "total_time"
            ])

    all_results = []
    total_start_time = time.time()
    algorithm_params = {
        "n_qubits_per_node": 2,
        "learning_rate": 0.01,
        "max_k": 20,
        "p": 1,
        "num_steps": 1000,
        "max_iter": 10,
        "adjacency_threshold": 0.3,
        "early_stop_threshold": 2,
        "penalty": 1000,
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
            print(f"Processing Graph {index}/{len(graphs)}: {base_title} (Standard-QAOA)")
            print(f"Graph Properties: {num_nodes} Nodes, {num_edges} Edges")
            print(f"{'='*50}")

            # 1. Original graph visualization (reuse from adapt function)
            try:
                plot_original_graph(graph, title=f"{base_title} - Original Graph (Nodes: {num_nodes}, Edges: {num_edges})")
            except Exception as e:
                handle_exception("plot_original_graph", index, e)

            # 2. 智能子图划分（限制量子比特数最多为21）
            max_qubits = 21
            subgraphs, sub_mappings, divide_info = smart_divide_graph_with_qubit_constraint(
                graph,
                max_qubits=max_qubits,
                max_k_per_subgraph=algorithm_params["max_k"],
                Q=algorithm_params["Q"]
            )
            print(f"Standard-QAOA 智能划分完成：{len(subgraphs)} 个子图（量子比特约束：≤{max_qubits}）")

            # 3. Subgraph visualization (optional)
            try:
                plot_New_IDs_subgraphs(subgraphs, sub_mappings, title=f"{base_title} - Subgraphs (Renumbered)")
                plot_Original_IDs_subgraphs(subgraphs, title=f"{base_title} - Subgraphs (Original IDs)")
            except Exception as e:
                handle_exception("subgraph plotting", index, e)

            # 4. 标准 QAOA 子图处理
            subgraph_start_time = time.time()
            subgraph_results = sequential_process_subgraphs_standard(
                subgraphs=subgraphs,
                sub_mappings=sub_mappings,
                dataset_name=dataset,
                graph_id=index,
                max_k=algorithm_params["max_k"],
                p=algorithm_params["p"],
                num_steps=algorithm_params["num_steps"],
                vertex_colors=None,
                nodes_to_recolor=None,
                penalty=algorithm_params["penalty"],
                Q=algorithm_params["Q"],
                learning_rate=algorithm_params["learning_rate"]
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

            # 5. 标准 QAOA 迭代优化
            optimized_coloring, opt_acc, conflict_counts, conflict_history, subgraph_opt_history = iterative_optimization_standard(
                graph=graph,
                subgraphs=subgraphs,
                sub_mappings=sub_mappings,
                subgraph_results=subgraph_results,
                max_k=algorithm_params["max_k"],
                p=algorithm_params["p"],
                num_steps=algorithm_params["num_steps"],
                max_iter=algorithm_params["max_iter"],
                adjacency_threshold=algorithm_params["adjacency_threshold"],
                early_stop_threshold=algorithm_params["early_stop_threshold"],
                penalty=algorithm_params["penalty"],
                Q=algorithm_params["Q"],
                learning_rate=algorithm_params["learning_rate"],
                vertex_colors=None,
                nodes_to_recolor=None,
                dataset_name=dataset,
                graph_id=index
            )

            # 6. 统一输出逻辑（与 adapt 完全一致）
            final_coloring = optimized_coloring
            unique_colors = len(set(final_coloring.values())) if final_coloring else 0
            final_conflicts = count_conflicts(final_coloring, graph) if final_coloring else -1
            reoptimization_count = sum(
                1 for h in subgraph_opt_history
                if isinstance(h, tuple) and len(h) >= 4 and h[3] > 0
            )
            min_k_list = [r[0] for r in subgraph_results if r is not None and r[0] is not None]
            best_k_value = min(unique_colors, max(min_k_list) if min_k_list else unique_colors)

            print(f"\n===== Optimization Summary (Standard-QAOA) =====")
            print(f"Final Conflicts: {final_conflicts} (Total Edges: {num_edges})")
            print(f"Final Accuracy: {opt_acc:.4f}")
            print(f"Colors Used: {unique_colors} (Global max_k limit: {algorithm_params['max_k']})")
            print(f"Best k Value: {best_k_value}")
            print(f"Subgraph Reoptimization Count: {reoptimization_count}")

            # 7. 子图着色可视化
            try:
                subgraph_colorings = [
                    get_subgraph_coloring(subgraph, final_coloring, mk)
                    for subgraph, mk in zip(subgraphs, min_k_list)
                ]
                plot_New_IDs_colored_subgraphs(
                    subgraphs, subgraph_colorings, sub_mappings, min_k_list,
                    title=f"{base_title} - 子图着色（新编号）", filename=base_title,
                    output_dir=os.path.join(BASE_DIR, "subgraph_visualizations")
                )
                plot_Original_IDs_colored_subgraphs(
                    subgraphs, subgraph_colorings,
                    title=f"{base_title} - 子图着色（原始编号）",
                    min_k_list=min_k_list, filename=base_title,
                    output_dir=os.path.join(BASE_DIR, "subgraph_visualizations")
                )
            except Exception as e:
                print(f"Error in colored subgraph plotting for graph {index}: {str(e)}")
                traceback.print_exc()

            # 8. 计算当前图的处理时间
            graph_time = time.time() - graph_start_time

            # 9. Final graph visualization
            try:
                final_graph_title = (
                    f"{base_title}\n"
                    f"Coloring Result (Colors: {unique_colors}, "
                    f"Nodes: {num_nodes}, Edges: {num_edges}, "
                    f"Conflicts: {final_conflicts})"
                )
                visualize_graph(
                    graph, coloring=final_coloring, title=final_graph_title,
                    index=index, min_k=unique_colors, filename=base_title,
                    processing_time=graph_time
                )
            except Exception as e:
                handle_exception("visualize_graph", index, e)

            # 10. 收集结果
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

            # 11. 写全局日志
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
                    round(time.time() - total_start_time, 4)
                ])

        except Exception as e:
            print(f"Uncaught exception while processing graph {index}: {e}")
            traceback.print_exc()
            continue

    # ---------- Post-processing ----------
    # 12. 计算总时间
    total_time = time.time() - total_start_time
    print(f"\n{'='*50}")
    print(f"Standard-QAOA all graphs processed, total time: {total_time:.1f}s")
    print(f"Successfully processed {len(all_results)}/{len(graphs)} graphs")
    print(f"CSV: {subgraph_csv} | Logs: {graph_log_csv}")
    print(f"{'='*50}")
    return all_results


# ---------- 含噪 QAOA 独立入口 ----------


def main_adapt_noise(graphs, dataset, graph_index, seed, depolarizing_prob=0.01):
    """
    含噪自适应QAOA主入口：完整流程与输出，支持退极化噪声模拟
    depolarizing_prob: 退极化噪声概率（0~1之间的浮点数，默认0.01）
    """
    # 验证噪声概率有效性
    if depolarizing_prob is None or not (0 <= depolarizing_prob <= 1):
        raise ValueError(f"Invalid noise probability: {depolarizing_prob}, must be a float between 0 and 1")

    # Create output directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    CSV_DIR = os.path.join(BASE_DIR, "csvs")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    # os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "graph_visualizations"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "subgraph_visualizations"), exist_ok=True)

    # Log file path (includes noise parameter identifier)
    noise_suffix = f"_noise_{depolarizing_prob:.3f}"
    subgraph_csv = os.path.join(CSV_DIR, f"adapt_noise_subgraph_results{noise_suffix}.csv")
    graph_log_csv = os.path.join(LOGS_DIR, f"adapt_noise_graph_results{noise_suffix}.log")

    # 初始化日志文件（若不存在）
    if not os.path.exists(subgraph_csv):
        with open(subgraph_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "subgraph_index",
                "nodes", "edges", "min_k", "conflicts", "status",
                "processing_time", "depolarizing_prob"
            ])
    if not os.path.exists(graph_log_csv):
        with open(graph_log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "nodes", "edges",
                "final_conflicts", "total_edges", "final_accuracy",
                "unique_colors", "global_max_k", "best_k_value",
                "subgraph_reoptimization_count", "processing_time",
                "conflict_changes", "total_time", "depolarizing_prob"
            ])

    all_results = []
    total_start = time.time()

    # Algorithm parameter configuration (includes noise parameter)
    algo_params = {
        "n_qubits_per_node": 2,
        "learning_rate": 0.01,
        "max_k": 20,
        "p": 3,  # QAOA layers
        "num_steps": 1000,
        "max_iter": 10,
        "adjacency_threshold": 0.3,
        "early_stop_threshold": 5,  # Relaxed early stop condition for noise scenario
        "penalty": 1000,
        "Q": 20,
        "depolarizing_prob": depolarizing_prob  # Depolarizing noise probability (ensure not None)
    }

    # Process each graph
    for idx, graph in enumerate(graphs, start=1):
        np.random.seed(seed + idx)  # Fix random seed for reproducibility
        g_start = time.time()
        try:
            # Get basic graph information
            g_name = getattr(graph, "file_name", f"graph_{idx}")
            base_ttl = os.path.splitext(g_name)[0]
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()

            print(f"\n{'='*50}")
            print(f"Noisy Adapt-QAOA | Graph {idx}/{len(graphs)}: {base_ttl}")
            print(f"Nodes: {n_nodes} | Edges: {n_edges} | Noise Probability: {depolarizing_prob}")
            print(f"{'='*50}")

            # 1. Draw original graph
            try:
                plot_original_graph(
                    graph,
                    title=f"[Noisy Adapt-QAOA] {base_ttl} - Original Graph "
                          f"(Nodes: {n_nodes}, Edges: {n_edges}, Noise: {depolarizing_prob})"
                )
            except Exception as e:
                handle_exception("plot_original_graph", idx, e)

            # 2. 智能子图划分（限制量子比特数最多为21）
            max_qubits = 21
            subgraphs, sub_maps, divide_info = smart_divide_graph_with_qubit_constraint(
                graph,
                max_qubits=max_qubits,
                max_k_per_subgraph=algo_params["max_k"],
                Q=algo_params["Q"]
            )
            print(f"Noisy Adapt-QAOA 智能划分完成：{len(subgraphs)} 个子图（量子比特约束：≤{max_qubits}）")

            # 3. Subgraph visualization
            try:
                plot_New_IDs_subgraphs(
                    subgraphs, sub_maps,
                    title=f"[Adapt-noise-QAOA] {base_ttl} - Subgraphs (Renumbered)"
                )
                plot_Original_IDs_subgraphs(
                    subgraphs,
                    title=f"[Adapt-noise-QAOA] {base_ttl} - Subgraphs (Original IDs)"
                )
            except Exception as e:
                handle_exception("subgraph plotting", idx, e)

            # 4. 含噪子图着色处理
            sub_start = time.time()
            sub_results = sequential_process_subgraphs_noise(
                subgraphs=subgraphs,
                sub_mappings=sub_maps,
                dataset_name=dataset,
                graph_id=idx,
                max_k=algo_params["max_k"],
                p=algo_params["p"],
                num_steps=algo_params["num_steps"],
                vertex_colors=None,
                nodes_to_recolor=None,
                penalty=algo_params["penalty"],
                Q=algo_params["Q"],
                learning_rate=algo_params["learning_rate"],
                depolarizing_prob=algo_params["depolarizing_prob"]  # 传递噪声参数
            )
            sub_time = time.time() - sub_start
            min_k_list = [r[0] for r in sub_results if r and r[0] is not None]

            # 写入子图日志
            dataset_name = os.path.basename(getattr(graph, "file_name", "unknown").split(os.sep)[0])
            for s_idx, res in enumerate(sub_results):
                if res is None:
                    continue
                mk, _, conf, stat, _ = res
                sg = subgraphs[s_idx] if s_idx < len(subgraphs) else None
                with open(subgraph_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        dataset_name, g_name, idx, s_idx + 1,
                        sg.number_of_nodes() if sg else 0,
                        sg.number_of_edges() if sg else 0,
                        mk, conf, stat,
                        round(sub_time / len(subgraphs), 4) if subgraphs else 0,
                        depolarizing_prob  # 记录噪声概率
                    ])

            # 5. 含噪迭代优化
            opt_color, opt_acc, conf_counts, conf_hist, sub_opt_hist = iterative_optimization_noise(
                graph=graph,
                subgraphs=subgraphs,
                sub_mappings=sub_maps,
                subgraph_results=sub_results,
                max_k=algo_params["max_k"],
                p=algo_params["p"],
                num_steps=algo_params["num_steps"],
                max_iter=algo_params["max_iter"],
                adjacency_threshold=algo_params["adjacency_threshold"],
                early_stop_threshold=algo_params["early_stop_threshold"],
                penalty=algo_params["penalty"],
                Q=algo_params["Q"],
                learning_rate=algo_params["learning_rate"],
                vertex_colors=None,
                nodes_to_recolor=None,
                dataset_name=dataset,
                graph_id=idx,
                depolarizing_prob=algo_params["depolarizing_prob"]  # 传递噪声参数
            )

            # 6. 结果统计
            final_color = opt_color
            uniq_colors = len(set(final_color.values())) if final_color else 0
            final_conf = count_conflicts(final_color, graph) if final_color else -1
            reopt_cnt = sum(1 for h in sub_opt_hist if isinstance(h, tuple) and len(h) >= 4 and h[3] > 0)
            best_k = min(uniq_colors, max(min_k_list) if min_k_list else uniq_colors)

            print(f"\n===== Adapt-noise-QAOA Optimization Summary =====")
            print(f"Final Conflicts: {final_conf} (Total Edges: {n_edges})")
            print(f"Final Accuracy: {opt_acc:.4f}")
            print(f"Colors Used: {uniq_colors}")
            print(f"Best k Value: {best_k}")
            print(f"Subgraph Reoptimization Count: {reopt_cnt}")
            print(f"Noise Parameter: Depolarizing Probability = {depolarizing_prob}")

            # 7. 子图着色可视化
            try:
                sub_colorings = [
                    get_subgraph_coloring(sg, final_color, mk)
                    for sg, mk in zip(subgraphs, min_k_list)
                ]
                plot_New_IDs_colored_subgraphs(
                    subgraphs, sub_colorings, sub_maps, min_k_list,
                    title=f"[Adapt-noise-QAOA] {base_ttl} - 子图着色（新编号）",
                    filename=f"Adapt_noise_{base_ttl}_p{depolarizing_prob:.3f}",
                    output_dir=os.path.join(BASE_DIR, "subgraph_visualizations")
                )
                plot_Original_IDs_colored_subgraphs(
                    subgraphs, sub_colorings,
                    title=f"[Adapt-noise-QAOA] {base_ttl} - 子图着色（原始编号）",
                    min_k_list=min_k_list,
                    filename=f"Adapt_noise_{base_ttl}_p{depolarizing_prob:.3f}",
                    output_dir=os.path.join(BASE_DIR, "subgraph_visualizations")
                )
            except Exception as e:
                print(f"Adapt-noise-QAOA subgraph coloring visualization failed: {e}")
                traceback.print_exc()

            # 8. 计算当前图的处理时间
            g_time = time.time() - g_start

            # 9. Final graph visualization
            try:
                vis_title = (
                    f"[Adapt-noise-QAOA] {base_ttl}\n"
                    f"coloring result(colors: {uniq_colors}, nodes: {n_nodes}, edges: {n_edges}, "
                    f"conflicts: {final_conf}, probability noise: {depolarizing_prob})"
                )
                visualize_graph(
                    graph, coloring=final_color, title=vis_title,
                    index=idx, min_k=uniq_colors,
                    filename=f"Adapt_noise_{base_ttl}_p{depolarizing_prob:.3f}",
                    processing_time=g_time
                )
            except Exception as e:
                handle_exception("visualize_graph", idx, e)

            # 10. 收集结果
            all_results.append({
                "graph_index": idx,
                "graph": graph,
                "final_coloring": final_color,
                "subgraphs": subgraphs,
                "sub_mappings": sub_maps,
                "subgraph_results": sub_results,
                "sub_colorings": sub_colorings,
                "conflict_counts": conf_counts,
                "conflict_history": conf_hist,
                "subgraph_opt_history": sub_opt_hist,
                "unique_colors": uniq_colors,
                "final_conflicts": final_conf,
                "accuracy": opt_acc,
                "processing_time": g_time,
                "num_nodes": n_nodes,
                "num_edges": n_edges,
                "base_title": base_ttl,
                "global_max_k": algo_params["max_k"],
                "best_k_value": best_k,
                "reoptimization_count": reopt_cnt,
                "noise_params": {"depolarizing_prob": depolarizing_prob}
            })

            # 11. 写入全局日志
            with open(graph_log_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    dataset_name, g_name, idx,
                    n_nodes, n_edges, final_conf, n_edges,
                    round(opt_acc, 4), uniq_colors,
                    algo_params["max_k"], best_k,
                    reopt_cnt, round(g_time, 4),
                    ",".join(map(str, conf_counts)) if conf_counts else "N/A",
                    round(time.time() - total_start, 4),
                    depolarizing_prob  # 记录噪声概率
                ])
            print(f"Adapt-noise-QAOA {idx} 完成，耗时: {g_time:.1f}秒")

        except Exception as e:
            print(f"Adapt-noise-QAOA processing graph {idx} exception: {e}")
            traceback.print_exc()
            continue

    # 12. 实验汇总
    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"Adapt-noise-QAOA all completed, total time: {total_time:.1f}s")
    print(f"Noise Parameter: Depolarizing Probability = {depolarizing_prob}")
    print(f"Successfully processed {len(all_results)}/{len(graphs)} graphs")
    print(f"CSV: {subgraph_csv} | Logs: {graph_log_csv}")
    print(f"{'='*50}")

    return all_results


# # ============ 使用示例 ============
# # Main_Multilevel_qaoa.py 使用示例
# python Main_Multilevel_qaoa.py --adapt --format-type col
# python Main_Multilevel_qaoa.py --standard --format-type pkl


# ============================================================================
# 主函数入口（指定具体数据执行着色可切换详细的run_experiments）
# ============================================================================

def main():
    """
    主函数：提供交互式菜单选择 QAOA 算法

    支持的算法:
    1. Adapt-QAOA: 自适应 QAOA
    2. Standard-QAOA: 标准 QAOA
    3. Adapt-QAOA with Noise: 含噪自适应 QAOA

    数据格式:
    - col: .col 格式图文件（位于 Data/instances/）
    - pkl: .pkl 格式图文件（位于 Data/instances/temp2/）
    - auto: 自动选择（优先 .col，否则 .pkl）
    """
    import sys

    print("\n" + "="*70)
    print("        QAOA Graph Coloring Algorithm Testing Tool")
    print("="*70)

    # Display menu
    print("\nPlease select the algorithm to run:")
    print("  1. Adapt-QAOA (Adaptive QAOA)")
    print("  2. Standard-QAOA (Standard QAOA)")
    print("  3. Adapt-QAOA with Noise (Noisy Adaptive QAOA)")
    print("  4. Run All Algorithms (Adapt + Standard + Noisy)")
    print("  0. Exit")

    try:
        choice = input("\nPlease enter option (0-4): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\nProgram exited")
        sys.exit(0)

    if choice == '0':
        print("Program exited")
        sys.exit(0)

    # Select data format
    print("\nPlease select data format:")
    print("  1. auto (Automatic selection: prioritize .col, otherwise .pkl)")
    print("  2. col (.col files only)")
    print("  3. pkl (.pkl files only)")
    try:
        format_choice = input("Please enter option (1-3, default=1): ").strip() or '1'
    except (EOFError, KeyboardInterrupt):
        print("\n\nProgram exited")
        sys.exit(0)

    format_map = {'1': 'auto', '2': 'col', '3': 'pkl'}
    format_type = format_map.get(format_choice, 'auto')

    # Select noise probability (only needed for noisy algorithm)
    noise_prob = 0.05
    if choice in ['3', '4']:
        try:
            noise_input = input(f"\nPlease enter noise probability (0-1, default=0.05): ").strip()
            if noise_input:
                noise_prob = float(noise_input)
                if not 0 <= noise_prob <= 1:
                    print("⚠️ Noise probability out of range, using default value 0.05")
                    noise_prob = 0.05
        except (EOFError, KeyboardInterrupt, ValueError):
            noise_prob = 0.05

    # Set experiment parameters
    seed = 10

    # Load graph data
    print("\n" + "="*70)
    print("Loading graph data...")
    print("="*70)
    graphs = load_graphs_from_dir('default', format_type=format_type)

    if not graphs:
        print("⚠️ No graph data loaded, program exiting")
        print(f"   Hint: Please ensure there are corresponding files in Data/instances/ or Data/instances/temp2/ directories")
        sys.exit(1)

    print(f'✓ Successfully loaded {len(graphs)} graphs (random seed: {seed})')

    # Store all algorithm results
    all_results = {}

    # 执行选中的算法
    if choice == '1' or choice == '4':
        # 运行 Adapt-QAOA
        print("\n" + "="*70)
        print("Running Adapt-QAOA...")
        print("="*70)
        try:
            adapt_results = main_adapt(graphs, dataset, 0, seed)
            all_results['Adapt-QAOA'] = adapt_results
            print(f"\n✅ Adapt-QAOA completed, successfully processed {len(adapt_results)} graphs")
        except Exception as e:
            print(f"\n⚠️ Adapt-QAOA execution failed: {e}")
            import traceback
            traceback.print_exc()

    if choice == '2' or choice == '4':
        # Run Standard-QAOA
        print("\n" + "="*70)
        print("Running Standard-QAOA...")
        print("="*70)
        try:
            standard_results = main_standard(graphs, dataset, 0, seed)
            all_results['Standard-QAOA'] = standard_results
            print(f"\n✅ Standard-QAOA completed, successfully processed {len(standard_results)} graphs")
        except Exception as e:
            print(f"\n⚠️ Standard-QAOA execution failed: {e}")
            import traceback
            traceback.print_exc()

    if choice == '3' or choice == '4':
        # Run Noisy Adapt-QAOA
        print("\n" + "="*70)
        print(f"Running Adapt-QAOA with Noise (Noise Probability: {noise_prob})...")
        print("="*70)
        try:
            noise_results = main_adapt_noise(graphs, dataset, 0, seed, depolarizing_prob=noise_prob)
            all_results['Adapt-QAOA-Noise'] = noise_results
            print(f"\n✅ Noisy Adapt-QAOA completed, successfully processed {len(noise_results)} graphs")
        except Exception as e:
            print(f"\n⚠️ Noisy Adapt-QAOA execution failed: {e}")
            import traceback
            traceback.print_exc()

    # Output results summary
    print("\n" + "="*70)
    print("                    Experiment Results Summary")
    print("="*70)

    for algo_name, results in all_results.items():
        if not results:
            continue

        print(f"\n【{algo_name}】")
        print("-" * 70)
        print(f"  Successfully Processed Graphs: {len(results)}")
        print(f"  {'Graph Index':<12} {'Colors':<10} {'Conflicts':<10} {'Accuracy':<12} {'Time(s)':<10}")
        print("  " + "-" * 60)

        for r in results:
            idx = r['graph_index']
            colors = r['unique_colors']
            conflicts = r['final_conflicts']
            accuracy = r['accuracy']
            time_cost = r['processing_time']
            print(f"  {idx:<12} {colors:<10} {conflicts:<10} {accuracy:<12.4f} {time_cost:<10.2f}")

        # Calculate statistics
        avg_colors = sum(r['unique_colors'] for r in results) / len(results)
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
        total_conflicts = sum(r['final_conflicts'] for r in results)

        print("  " + "-" * 60)
        print(f"  Average Colors: {avg_colors:.2f}")
        print(f"  Average Time: {avg_time:.2f} s")
        print(f"  Average Accuracy: {avg_accuracy:.4f}")
        print(f"  Total Conflicts: {total_conflicts}")

    # Comparison summary (if multiple algorithms were run)
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("                    Algorithm Comparison")
        print("="*70)
        print(f"  {'Algorithm Name':<20} {'Avg Colors':<15} {'Avg Accuracy':<15} {'Avg Time(s)':<15}")
        print("  " + "-" * 65)

        for algo_name, results in all_results.items():
            if results:
                avg_colors = sum(r['unique_colors'] for r in results) / len(results)
                avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
                avg_time = sum(r['processing_time'] for r in results) / len(results)
                print(f"  {algo_name:<20} {avg_colors:<15.2f} {avg_accuracy:<15.4f} {avg_time:<15.2f}")

    print("\n" + "="*70)
    print("Experiment completed! Log files have been saved to logs/ directory")
    print("="*70 + "\n")


# ========================================================================
# 大规模数据集处理函数
# ========================================================================

def load_large_datasets():
    """加载大规模数据集"""
    # 数据目录: HAdaQAOA/Data/Large_datesets/
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "Data", "Large_datesets")

    large_datasets = [
        # "cora.col",
        "citeseer.col",
        "pubmed.col"
    ]

    print(f"\n加载大规模图数据...")
    print(f"  数据目录: {DATA_DIR}")

    graphs = []
    for dataset_name in large_datasets:
        dataset_path = os.path.join(DATA_DIR, dataset_name)
        if os.path.exists(dataset_path):
            print(f"  正在加载: {dataset_name}")
            try:
                graph = read_col_file(dataset_path)
                if graph is not None:
                    graph.file_name = dataset_name
                    graphs.append(graph)
                    print(f"    ✓ 成功加载 {dataset_name}: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
                else:
                    print(f"    ✗ 加载失败: {dataset_name}")
            except Exception as e:
                print(f"    ✗ 加载 {dataset_name} 时出错: {e}")
                traceback.print_exc()
        else:
            print(f"  ✗ 文件不存在: {dataset_path}")

    return graphs


def print_large_dataset_analysis(graphs, max_nodes_per_subgraph):
    """打印大规模数据集分析"""
    print(f"\n📊 大规模图数据分析:")
    print(f"{'图名':<15} {'节点':<8} {'边':<10} {'边密度':<12} {'可行性':<8} {'推荐子图数'}")
    print("─" * 75)

    for i, g in enumerate(graphs, 1):
        g_name = getattr(g, 'file_name', f'graph_{i}')
        base_name = os.path.splitext(g_name)[0]

        validation = validate_graph_feasibility(g, max_nodes_per_subgraph)
        feasible_str = "✓ 可行" if validation['is_feasible'] else "✗ 需划分"

        print(f"{base_name:<15} {validation['num_nodes']:<8} {validation['num_edges']:<10} "
              f"{validation['edge_density']:<12.6f} {feasible_str:<8} "
              f"{validation['recommended_subgraphs']}")
    print("─" * 75)


def main_adapt_large(graphs, dataset, graph_index, seed, run_classical=False):
    """
    Adapt-QAOA 大规模数据集入口（加载大规模数据，使用智能划分）
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    CSV_DIR = os.path.join(BASE_DIR, "csvs")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "large_graph_visualizations"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "large_subgraph_visualizations"), exist_ok=True)

    subgraph_csv = os.path.join(CSV_DIR, "large_datasets_adapt_subgraph_results.csv")
    if not os.path.exists(subgraph_csv):
        with open(subgraph_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "subgraph_index",
                "nodes", "edges", "min_k", "conflicts", "status", "processing_time"
            ])

    graph_log_csv = os.path.join(LOGS_DIR, "large_datasets_adapt_graph_results.log")
    if not os.path.exists(graph_log_csv):
        with open(graph_log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "nodes", "edges",
                "final_conflicts", "total_edges", "final_accuracy",
                "unique_colors", "global_max_k", "best_k_value",
                "subgraph_reoptimization_count", "processing_time",
                "conflict_changes", "total_time"
            ])

    # 创建经典算法结果CSV
    classical_csv = os.path.join(CSV_DIR, "large_datasets_adapt_classical_results.csv")
    if run_classical and not os.path.exists(classical_csv):
        with open(classical_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "algorithm",
                "num_nodes", "num_edges", "num_colors", "conflicts",
                "is_valid", "execution_time_ms"
            ])

    # 配置日志
    log_file = os.path.join(LOGS_DIR, "large_adapt_qaoa.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    all_results = []
    total_start_time = time.time()

    # 算法参数配置 - 针对大规模数据集优化
    algorithm_params = {
        "max_k": 8,
        "p": 3,
        "num_steps": 500,
        "max_iter": 15,
        "early_stop_threshold": 5,
        "penalty": 1000,
        "Q": 20
    }

    for index, graph in enumerate(graphs, start=1):
        graph_start_time = time.time()
        try:
            graph_name = getattr(graph, "file_name", f"graph_{index}")
            base_title = os.path.splitext(graph_name)[0]
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()

            print(f"\n{'='*60}")
            print(f"Processing Graph {index}/{len(graphs)}: {base_title} (Adapt-QAOA Large)")
            print(f"Graph Properties: {num_nodes} Nodes, {num_edges} Edges")
            print(f"{'='*60}")

            # 0. 运行经典算法对比（如果启用）
            classical_results = {}
            if run_classical:
                print(f"\n{'─'*50}")
                print(f"Running Classical Algorithms Comparison")
                print(f"{'─'*50}")

                greedy_res = run_greedy_coloring(graph, graph_name)
                if greedy_res:
                    print(f"✓ Greedy Algorithm:")
                    print(f"  Colors: {greedy_res['num_colors']}, Conflicts: {greedy_res['conflicts']}, "
                          f"Time: {greedy_res['execution_time_ms']:.2f}ms")
                    classical_results['greedy'] = greedy_res

                    dataset_name = os.path.basename(getattr(graph, "file_name", "unknown").split(os.sep)[0])
                    with open(classical_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            dataset_name, graph_name, index, greedy_res['algorithm'],
                            greedy_res['num_nodes'], greedy_res['num_edges'],
                            greedy_res['num_colors'], greedy_res['conflicts'],
                            greedy_res['is_valid'], greedy_res['execution_time_ms']
                        ])

                wp_res = run_welch_powell_coloring(graph, graph_name)
                if wp_res:
                    print(f"✓ Welch-Powell Algorithm:")
                    print(f"  Colors: {wp_res['num_colors']}, Conflicts: {wp_res['conflicts']}, "
                          f"Time: {wp_res['execution_time_ms']:.2f}ms")
                    classical_results['welch_powell'] = wp_res

                    with open(classical_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            dataset_name, graph_name, index, wp_res['algorithm'],
                            wp_res['num_nodes'], wp_res['num_edges'],
                            wp_res['num_colors'], wp_res['conflicts'],
                            wp_res['is_valid'], wp_res['execution_time_ms']
                        ])

            # 1. 原始图可视化
            try:
                filename = f"{base_title}_original"
                plot_original_graph(
                    graph,
                    title=f"[Adapt-QAOA] {base_title} - Original Graph (Nodes: {num_nodes}, Edges: {num_edges})",
                    filename=filename,
                    output_dir=os.path.join(BASE_DIR, "large_graph_visualizations")
                )
            except Exception as e:
                handle_exception("plot_original_graph", index, e)

            # 2. 智能子图划分（限制量子比特数最多为21）
            max_qubits = 21
            subgraphs, sub_mappings, divide_info = smart_divide_graph_with_qubit_constraint(
                graph,
                max_qubits=max_qubits,
                max_k_per_subgraph=algorithm_params["max_k"],
                Q=algorithm_params["Q"]
            )
            logger.info(f"Adapt-QAOA 智能划分完成: {len(subgraphs)} 个子图（量子比特约束：≤{max_qubits}）")

            # 3. 子图可视化
            try:
                plot_New_IDs_subgraphs(
                    subgraphs, sub_mappings,
                    title=f"[Adapt-QAOA] {base_title} - Subgraphs (Renumbered)",
                    filename=f"{base_title}_subgraphs_renumbered",
                    output_dir=os.path.join(BASE_DIR, "large_subgraph_visualizations")
                )
                plot_Original_IDs_subgraphs(
                    subgraphs,
                    title=f"[Adapt-QAOA] {base_title} - Subgraphs (Original IDs)",
                    filename=f"{base_title}_subgraphs_original",
                    output_dir=os.path.join(BASE_DIR, "large_subgraph_visualizations")
                )
            except Exception as e:
                handle_exception("subgraph plotting", index, e)

            # 4. Adapt-QAOA 子图处理
            subgraph_start_time = time.time()
            subgraph_results = sequential_process_subgraphs(
                subgraphs=subgraphs,
                sub_mappings=sub_mappings,
                dataset_name=dataset,
                graph_id=index,
                max_k=algorithm_params["max_k"],
                p=algorithm_params["p"],
                num_steps=algorithm_params["num_steps"],
                vertex_colors=None,
                nodes_to_recolor=None,
                penalty=algorithm_params["penalty"],
                Q=algorithm_params["Q"],
                learning_rate=0.01
            )
            subgraph_total_time = time.time() - subgraph_start_time

            # 记录子图级别结果
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

            # 5. Adapt-QAOA 迭代优化
            opt_color, opt_acc, conflict_counts, conflict_history, sub_opt_hist = iterative_optimization(
                graph=graph,
                subgraphs=subgraphs,
                sub_mappings=sub_mappings,
                subgraph_results=subgraph_results,
                max_k=algorithm_params["max_k"],
                p=algorithm_params["p"],
                num_steps=algorithm_params["num_steps"],
                max_iter=algorithm_params["max_iter"],
                adjacency_threshold=0.3,
                early_stop_threshold=algorithm_params["early_stop_threshold"],
                penalty=algorithm_params["penalty"],
                Q=algorithm_params["Q"],
                learning_rate=0.01,
                vertex_colors=None,
                nodes_to_recolor=None,
                dataset_name=dataset,
                graph_id=index
            )

            # 6. 统计结果
            final_coloring = opt_color
            unique_colors = len(set(final_coloring.values())) if final_coloring else 0
            final_conflicts = count_conflicts(final_coloring, graph) if final_coloring else -1
            reoptimization_count = sum(
                1 for h in sub_opt_hist
                if isinstance(h, tuple) and len(h) >= 4 and h[3] > 0
            )
            min_k_list = [r[0] for r in subgraph_results if r is not None and r[0] is not None]
            best_k_value = min(unique_colors, max(min_k_list) if min_k_list else unique_colors)

            print(f"\n===== Optimization Summary (Adapt-QAOA Large) =====")
            print(f"Final Conflicts: {final_conflicts} (Total Edges: {num_edges})")
            print(f"Final Accuracy: {opt_acc:.4f}")
            print(f"Colors Used: {unique_colors} (Global max_k limit: {algorithm_params['max_k']})")
            print(f"Best k Value: {best_k_value}")

            # 经典算法对比输出
            if run_classical and classical_results:
                print(f"\n{'─'*50}")
                print(f"Classical Algorithms Comparison")
                print(f"{'─'*50}")

                graph_time = time.time() - graph_start_time
                qaoa_time_ms = graph_time * 1000
                qaoa_valid = "Yes" if final_conflicts == 0 else "No"

                print(f"{'Algorithm':<20} {'Colors':<10} {'Conflicts':<10} {'Time (ms)':<15} {'Valid'}")
                print(f"{'─'*65}")
                print(f"{'QAOA-Adapt':<20} {unique_colors:<10} {final_conflicts:<10} {qaoa_time_ms:<15.2f} {qaoa_valid}")

                if 'greedy' in classical_results:
                    greedy = classical_results['greedy']
                    greedy_valid = "Yes" if greedy['conflicts'] == 0 else "No"
                    print(f"{'Greedy':<20} {greedy['num_colors']:<10} {greedy['conflicts']:<10} "
                          f"{greedy['execution_time_ms']:<15.2f} {greedy_valid}")

                if 'welch_powell' in classical_results:
                    wp = classical_results['welch_powell']
                    wp_valid = "Yes" if wp['conflicts'] == 0 else "No"
                    print(f"{'Welch-Powell':<20} {wp['num_colors']:<10} {wp['conflicts']:<10} "
                          f"{wp['execution_time_ms']:<15.2f} {wp_valid}")
                print(f"{'─'*65}")

            # 7. 子图着色可视化
            try:
                subgraph_colorings = [
                    get_subgraph_coloring(subgraph, final_coloring, mk)
                    for subgraph, mk in zip(subgraphs, min_k_list)
                ]
                plot_New_IDs_colored_subgraphs(
                    subgraphs, subgraph_colorings, sub_mappings, min_k_list,
                    title=f"[Adapt-QAOA] {base_title} - Colored Subgraphs (Renumbered)",
                    filename=f"{base_title}_colored_subgraphs_renumbered",
                    output_dir=os.path.join(BASE_DIR, "large_subgraph_visualizations")
                )
                plot_Original_IDs_colored_subgraphs(
                    subgraphs, subgraph_colorings,
                    title=f"[Adapt-QAOA] {base_title} - Colored Subgraphs (Original IDs)",
                    min_k_list=min_k_list, filename=f"{base_title}_colored_subgraphs_original",
                    output_dir=os.path.join(BASE_DIR, "large_subgraph_visualizations")
                )
            except Exception as e:
                print(f"Error in colored subgraph plotting for graph {index}: {str(e)}")
                traceback.print_exc()

            # 8. 最终图可视化
            graph_time = time.time() - graph_start_time
            try:
                final_graph_title = (
                    f"[Adapt-QAOA] {base_title}\n"
                    f"Coloring Result (Colors: {unique_colors}, "
                    f"Nodes: {num_nodes}, Edges: {num_edges}, "
                    f"Conflicts: {final_conflicts})"
                )
                visualize_graph(
                    graph, coloring=final_coloring, title=final_graph_title,
                    index=index, min_k=unique_colors,
                    filename=f"adapt_{base_title}_final_coloring",
                    output_dir=os.path.join(BASE_DIR, "large_graph_visualizations"),
                    processing_time=graph_time
                )
            except Exception as e:
                handle_exception("visualize_graph", index, e)

            # 9. 收集结果
            result = {
                "graph_index": index,
                "graph": graph,
                "final_coloring": final_coloring,
                "subgraphs": subgraphs,
                "sub_mappings": sub_mappings,
                "subgraph_results": subgraph_results,
                "sub_colorings": subgraph_colorings if 'subgraph_colorings' in locals() else [],
                "conflict_counts": conflict_counts,
                "conflict_history": conflict_history,
                "subgraph_opt_history": sub_opt_hist,
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

            # 10. 写全局日志
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
                    round(time.time() - total_start_time, 4)
                ])

        except Exception as e:
            print(f"Uncaught exception while processing graph {index}: {e}")
            traceback.print_exc()
            continue

    # 结果汇总
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"Large-scale Adapt-QAOA all graphs processed, total time: {total_time:.1f}s")
    print(f"Successfully processed {len(all_results)}/{len(graphs)} graphs")
    print(f"CSV: {subgraph_csv} | Logs: {graph_log_csv}")
    print(f"{'='*60}")
    return all_results


def main_adapt_noise_large(graphs, dataset, graph_index, seed, run_classical=False, depolarizing_prob=0.05):
    """
    Noisy Adapt-QAOA 大规模数据集入口（加载大规模数据，使用智能划分）
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    CSV_DIR = os.path.join(BASE_DIR, "csvs")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "large_graph_visualizations"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "large_subgraph_visualizations"), exist_ok=True)

    # Log file path (includes noise parameter identifier)
    noise_suffix = f"_noise_{depolarizing_prob:.3f}"
    subgraph_csv = os.path.join(CSV_DIR, f"large_datasets_adapt_noise_subgraph_results{noise_suffix}.csv")
    if not os.path.exists(subgraph_csv):
        with open(subgraph_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "subgraph_index",
                "nodes", "edges", "min_k", "conflicts", "status",
                "processing_time", "depolarizing_prob"
            ])

    graph_log_csv = os.path.join(LOGS_DIR, f"large_datasets_adapt_noise_graph_results{noise_suffix}.log")
    if not os.path.exists(graph_log_csv):
        with open(graph_log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "nodes", "edges",
                "final_conflicts", "total_edges", "final_accuracy",
                "unique_colors", "global_max_k", "best_k_value",
                "subgraph_reoptimization_count", "processing_time",
                "conflict_changes", "total_time", "depolarizing_prob"
            ])

    # 创建经典算法结果CSV
    classical_csv = os.path.join(CSV_DIR, f"large_datasets_adapt_noise_classical_results{noise_suffix}.csv")
    if run_classical and not os.path.exists(classical_csv):
        with open(classical_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "algorithm",
                "num_nodes", "num_edges", "num_colors", "conflicts",
                "is_valid", "execution_time_ms"
            ])

    # 配置日志
    log_file = os.path.join(LOGS_DIR, f"large_adapt_noise_qaoa{noise_suffix}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    all_results = []
    total_start_time = time.time()

    # 算法参数配置 - 针对大规模数据集优化
    algorithm_params = {
        "max_k": 8,
        "p": 3,
        "num_steps": 500,
        "max_iter": 15,
        "early_stop_threshold": 5,
        "penalty": 1000,
        "Q": 20,
        "depolarizing_prob": depolarizing_prob
    }

    for index, graph in enumerate(graphs, start=1):
        graph_start_time = time.time()
        try:
            graph_name = getattr(graph, "file_name", f"graph_{index}")
            base_title = os.path.splitext(graph_name)[0]
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()

            print(f"\n{'='*60}")
            print(f"Processing Graph {index}/{len(graphs)}: {base_title} (Noisy Adapt-QAOA Large, p={depolarizing_prob})")
            print(f"Graph Properties: {num_nodes} Nodes, {num_edges} Edges")
            print(f"{'='*60}")

            # 0. 运行经典算法对比（如果启用）
            classical_results = {}
            if run_classical:
                print(f"\n{'─'*50}")
                print(f"Running Classical Algorithms Comparison")
                print(f"{'─'*50}")

                greedy_res = run_greedy_coloring(graph, graph_name)
                if greedy_res:
                    print(f"✓ Greedy Algorithm:")
                    print(f"  Colors: {greedy_res['num_colors']}, Conflicts: {greedy_res['conflicts']}, "
                          f"Time: {greedy_res['execution_time_ms']:.2f}ms")
                    classical_results['greedy'] = greedy_res

                    dataset_name = os.path.basename(getattr(graph, "file_name", "unknown").split(os.sep)[0])
                    with open(classical_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            dataset_name, graph_name, index, greedy_res['algorithm'],
                            greedy_res['num_nodes'], greedy_res['num_edges'],
                            greedy_res['num_colors'], greedy_res['conflicts'],
                            greedy_res['is_valid'], greedy_res['execution_time_ms']
                        ])

                wp_res = run_welch_powell_coloring(graph, graph_name)
                if wp_res:
                    print(f"✓ Welch-Powell Algorithm:")
                    print(f"  Colors: {wp_res['num_colors']}, Conflicts: {wp_res['conflicts']}, "
                          f"Time: {wp_res['execution_time_ms']:.2f}ms")
                    classical_results['welch_powell'] = wp_res

                    with open(classical_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            dataset_name, graph_name, index, wp_res['algorithm'],
                            wp_res['num_nodes'], wp_res['num_edges'],
                            wp_res['num_colors'], wp_res['conflicts'],
                            wp_res['is_valid'], wp_res['execution_time_ms']
                        ])

            # 1. 原始图可视化
            try:
                filename = f"{base_title}_original"
                plot_original_graph(
                    graph,
                    title=f"[Noisy Adapt-QAOA] {base_title} - Original Graph (Nodes: {num_nodes}, Edges: {num_edges}, Noise: {depolarizing_prob})",
                    filename=filename,
                    output_dir=os.path.join(BASE_DIR, "large_graph_visualizations")
                )
            except Exception as e:
                handle_exception("plot_original_graph", index, e)

            # 2. 智能子图划分（限制量子比特数最多为21）
            max_qubits = 21
            subgraphs, sub_mappings, divide_info = smart_divide_graph_with_qubit_constraint(
                graph,
                max_qubits=max_qubits,
                max_k_per_subgraph=algorithm_params["max_k"],
                Q=algorithm_params["Q"]
            )
            logger.info(f"Noisy Adapt-QAOA 智能划分完成: {len(subgraphs)} 个子图（量子比特约束：≤{max_qubits}）")

            # 3. 子图可视化
            try:
                plot_New_IDs_subgraphs(
                    subgraphs, sub_mappings,
                    title=f"[Noisy Adapt-QAOA] {base_title} - Subgraphs (Renumbered)",
                    filename=f"{base_title}_subgraphs_renumbered",
                    output_dir=os.path.join(BASE_DIR, "large_subgraph_visualizations")
                )
                plot_Original_IDs_subgraphs(
                    subgraphs,
                    title=f"[Noisy Adapt-QAOA] {base_title} - Subgraphs (Original IDs)",
                    filename=f"{base_title}_subgraphs_original",
                    output_dir=os.path.join(BASE_DIR, "large_subgraph_visualizations")
                )
            except Exception as e:
                handle_exception("subgraph plotting", index, e)

            # 4. Noisy Adapt-QAOA 子图处理
            subgraph_start_time = time.time()
            subgraph_results = sequential_process_subgraphs_noise(
                subgraphs=subgraphs,
                sub_mappings=sub_mappings,
                dataset_name=dataset,
                graph_id=index,
                max_k=algorithm_params["max_k"],
                p=algorithm_params["p"],
                num_steps=algorithm_params["num_steps"],
                vertex_colors=None,
                nodes_to_recolor=None,
                penalty=algorithm_params["penalty"],
                Q=algorithm_params["Q"],
                learning_rate=0.01,
                depolarizing_prob=algorithm_params["depolarizing_prob"]
            )
            subgraph_total_time = time.time() - subgraph_start_time

            # 记录子图级别结果
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
                        round(subgraph_total_time / len(subgraphs), 4) if subgraphs else 0,
                        depolarizing_prob
                    ])

            # 5. Noisy Adapt-QAOA 迭代优化
            opt_color, opt_acc, conflict_counts, conflict_history, sub_opt_hist = iterative_optimization_noise(
                graph=graph,
                subgraphs=subgraphs,
                sub_mappings=sub_mappings,
                subgraph_results=subgraph_results,
                max_k=algorithm_params["max_k"],
                p=algorithm_params["p"],
                num_steps=algorithm_params["num_steps"],
                max_iter=algorithm_params["max_iter"],
                adjacency_threshold=0.3,
                early_stop_threshold=algorithm_params["early_stop_threshold"],
                penalty=algorithm_params["penalty"],
                Q=algorithm_params["Q"],
                learning_rate=0.01,
                vertex_colors=None,
                nodes_to_recolor=None,
                dataset_name=dataset,
                graph_id=index,
                depolarizing_prob=algorithm_params["depolarizing_prob"]
            )

            # 6. 统计结果
            final_coloring = opt_color
            unique_colors = len(set(final_coloring.values())) if final_coloring else 0
            final_conflicts = count_conflicts(final_coloring, graph) if final_coloring else -1
            reoptimization_count = sum(
                1 for h in sub_opt_hist
                if isinstance(h, tuple) and len(h) >= 4 and h[3] > 0
            )
            min_k_list = [r[0] for r in subgraph_results if r is not None and r[0] is not None]
            best_k_value = min(unique_colors, max(min_k_list) if min_k_list else unique_colors)

            print(f"\n===== Optimization Summary (Noisy Adapt-QAOA Large) =====")
            print(f"Final Conflicts: {final_conflicts} (Total Edges: {num_edges})")
            print(f"Final Accuracy: {opt_acc:.4f}")
            print(f"Colors Used: {unique_colors} (Global max_k limit: {algorithm_params['max_k']})")
            print(f"Best k Value: {best_k_value}")
            print(f"Noise Parameter: Depolarizing Probability = {depolarizing_prob}")

            # 经典算法对比输出
            if run_classical and classical_results:
                print(f"\n{'─'*50}")
                print(f"Classical Algorithms Comparison")
                print(f"{'─'*50}")

                graph_time = time.time() - graph_start_time
                qaoa_time_ms = graph_time * 1000
                qaoa_valid = "Yes" if final_conflicts == 0 else "No"

                print(f"{'Algorithm':<20} {'Colors':<10} {'Conflicts':<10} {'Time (ms)':<15} {'Valid'}")
                print(f"{'─'*65}")
                print(f"{'QAOA-Adapt-Noise':<20} {unique_colors:<10} {final_conflicts:<10} {qaoa_time_ms:<15.2f} {qaoa_valid}")

                if 'greedy' in classical_results:
                    greedy = classical_results['greedy']
                    greedy_valid = "Yes" if greedy['conflicts'] == 0 else "No"
                    print(f"{'Greedy':<20} {greedy['num_colors']:<10} {greedy['conflicts']:<10} "
                          f"{greedy['execution_time_ms']:<15.2f} {greedy_valid}")

                if 'welch_powell' in classical_results:
                    wp = classical_results['welch_powell']
                    wp_valid = "Yes" if wp['conflicts'] == 0 else "No"
                    print(f"{'Welch-Powell':<20} {wp['num_colors']:<10} {wp['conflicts']:<10} "
                          f"{wp['execution_time_ms']:<15.2f} {wp_valid}")
                print(f"{'─'*65}")

            # 7. 子图着色可视化
            try:
                subgraph_colorings = [
                    get_subgraph_coloring(subgraph, final_coloring, mk)
                    for subgraph, mk in zip(subgraphs, min_k_list)
                ]
                plot_New_IDs_colored_subgraphs(
                    subgraphs, subgraph_colorings, sub_mappings, min_k_list,
                    title=f"[Noisy Adapt-QAOA] {base_title} - Colored Subgraphs (Renumbered)",
                    filename=f"{base_title}_colored_subgraphs_renumbered",
                    output_dir=os.path.join(BASE_DIR, "large_subgraph_visualizations")
                )
                plot_Original_IDs_colored_subgraphs(
                    subgraphs, subgraph_colorings,
                    title=f"[Noisy Adapt-QAOA] {base_title} - Colored Subgraphs (Original IDs)",
                    min_k_list=min_k_list, filename=f"{base_title}_colored_subgraphs_original",
                    output_dir=os.path.join(BASE_DIR, "large_subgraph_visualizations")
                )
            except Exception as e:
                print(f"Error in colored subgraph plotting for graph {index}: {str(e)}")
                traceback.print_exc()

            # 8. 最终图可视化
            graph_time = time.time() - graph_start_time
            try:
                final_graph_title = (
                    f"[Noisy Adapt-QAOA] {base_title}\n"
                    f"Coloring Result (Colors: {unique_colors}, "
                    f"Nodes: {num_nodes}, Edges: {num_edges}, "
                    f"Conflicts: {final_conflicts}, Noise: {depolarizing_prob})"
                )
                visualize_graph(
                    graph, coloring=final_coloring, title=final_graph_title,
                    index=index, min_k=unique_colors,
                    filename=f"adapt_noise_{base_title}_final_coloring",
                    output_dir=os.path.join(BASE_DIR, "large_graph_visualizations"),
                    processing_time=graph_time
                )
            except Exception as e:
                handle_exception("visualize_graph", index, e)

            # 9. 收集结果
            result = {
                "graph_index": index,
                "graph": graph,
                "final_coloring": final_coloring,
                "subgraphs": subgraphs,
                "sub_mappings": sub_mappings,
                "subgraph_results": subgraph_results,
                "sub_colorings": subgraph_colorings if 'subgraph_colorings' in locals() else [],
                "conflict_counts": conflict_counts,
                "conflict_history": conflict_history,
                "subgraph_opt_history": sub_opt_hist,
                "unique_colors": unique_colors,
                "final_conflicts": final_conflicts,
                "accuracy": opt_acc,
                "processing_time": graph_time,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "base_title": base_title,
                "global_max_k": algorithm_params["max_k"],
                "best_k_value": best_k_value,
                "reoptimization_count": reoptimization_count,
                "noise_params": {"depolarizing_prob": depolarizing_prob}
            }
            all_results.append(result)

            # 10. 写全局日志
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
                    depolarizing_prob
                ])

        except Exception as e:
            print(f"Uncaught exception while processing graph {index}: {e}")
            traceback.print_exc()
            continue

    # 结果汇总
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"Large-scale Noisy Adapt-QAOA all graphs processed, total time: {total_time:.1f}s")
    print(f"Noise Parameter: Depolarizing Probability = {depolarizing_prob}")
    print(f"Successfully processed {len(all_results)}/{len(graphs)} graphs")
    print(f"CSV: {subgraph_csv} | Logs: {graph_log_csv}")
    print(f"{'='*60}")
    return all_results


def main_standard_large(graphs, dataset, graph_index, seed, run_classical=False):
    """
    标准 QAOA 大规模数据集入口
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    CSV_DIR = os.path.join(BASE_DIR, "csvs")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "large_graph_visualizations"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "large_subgraph_visualizations"), exist_ok=True)

    subgraph_csv = os.path.join(CSV_DIR, "large_datasets_standard_subgraph_results.csv")
    if not os.path.exists(subgraph_csv):
        with open(subgraph_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "subgraph_index",
                "nodes", "edges", "min_k", "conflicts", "status", "processing_time"
            ])

    graph_log_csv = os.path.join(LOGS_DIR, "large_datasets_standard_graph_results.log")
    if not os.path.exists(graph_log_csv):
        with open(graph_log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "nodes", "edges",
                "final_conflicts", "total_edges", "final_accuracy",
                "unique_colors", "global_max_k", "best_k_value",
                "subgraph_reoptimization_count", "processing_time",
                "conflict_changes", "total_time"
            ])

    # 创建经典算法结果CSV
    classical_csv = os.path.join(CSV_DIR, "large_datasets_standard_classical_results.csv")
    if run_classical and not os.path.exists(classical_csv):
        with open(classical_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "dataset", "graph_name", "graph_index", "algorithm",
                "num_nodes", "num_edges", "num_colors", "conflicts",
                "is_valid", "execution_time_ms"
            ])

    # 配置日志
    log_file = os.path.join(LOGS_DIR, "large_standard_qaoa.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    all_results = []
    total_start_time = time.time()

    # 算法参数配置 - 针对大规模数据集优化
    algorithm_params = {
        "max_k": 8,  # 降低max_k以提高效率
        "p": 1,
        "num_steps": 500,  # 减少步数
        "max_iter": 15,
        "early_stop_threshold": 5,
        "penalty": 1000,
        "Q": 20
    }

    # 处理每张图
    for index, graph in enumerate(graphs, start=1):
        graph_start_time = time.time()
        try:
            graph_name = getattr(graph, "file_name", f"graph_{index}")
            base_title = os.path.splitext(graph_name)[0]
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()

            print(f"\n{'='*60}")
            print(f"Processing Graph {index}/{len(graphs)}: {base_title} (Large-Scale)")
            print(f"Graph Properties: {num_nodes} Nodes, {num_edges} Edges")
            print(f"{'='*60}")

            # 0. 运行经典算法对比（如果启用）
            classical_results = {}
            if run_classical:
                print(f"\n{'─'*50}")
                print(f"Running Classical Algorithms Comparison")
                print(f"{'─'*50}")

                # 贪心算法
                greedy_res = run_greedy_coloring(graph, graph_name)
                if greedy_res:
                    print(f"✓ Greedy Algorithm:")
                    print(f"  Colors: {greedy_res['num_colors']}, Conflicts: {greedy_res['conflicts']}, "
                          f"Time: {greedy_res['execution_time_ms']:.2f}ms")
                    classical_results['greedy'] = greedy_res

                    # 保存到CSV
                    dataset_name = os.path.basename(getattr(graph, "file_name", "unknown").split(os.sep)[0])
                    with open(classical_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            dataset_name, graph_name, index, greedy_res['algorithm'],
                            greedy_res['num_nodes'], greedy_res['num_edges'],
                            greedy_res['num_colors'], greedy_res['conflicts'],
                            greedy_res['is_valid'], greedy_res['execution_time_ms']
                        ])

                # Welch-Powell算法
                wp_res = run_welch_powell_coloring(graph, graph_name)
                if wp_res:
                    print(f"✓ Welch-Powell Algorithm:")
                    print(f"  Colors: {wp_res['num_colors']}, Conflicts: {wp_res['conflicts']}, "
                          f"Time: {wp_res['execution_time_ms']:.2f}ms")
                    classical_results['welch_powell'] = wp_res

                    # 保存到CSV
                    with open(classical_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            dataset_name, graph_name, index, wp_res['algorithm'],
                            wp_res['num_nodes'], wp_res['num_edges'],
                            wp_res['num_colors'], wp_res['conflicts'],
                            wp_res['is_valid'], wp_res['execution_time_ms']
                        ])

            # 1. 原始图可视化
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

            # 2. 智能子图划分（限制量子比特数最多为21）
            max_qubits = 21
            subgraphs, sub_mappings, divide_info = smart_divide_graph_with_qubit_constraint(
                graph,
                max_qubits=max_qubits,
                max_k_per_subgraph=algorithm_params["max_k"],
                Q=algorithm_params["Q"]
            )
            logger.info(f"智能划分完成: {len(subgraphs)} 个子图（量子比特约束：≤{max_qubits}）")

            # 3. 子图可视化
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

            # 4. 标准 QAOA 子图处理
            subgraph_start_time = time.time()
            subgraph_results = sequential_process_subgraphs_standard(
                subgraphs=subgraphs,
                sub_mappings=sub_mappings,
                dataset_name=dataset,
                graph_id=index,
                max_k=algorithm_params["max_k"],
                p=algorithm_params["p"],
                num_steps=algorithm_params["num_steps"],
                vertex_colors=None,
                nodes_to_recolor=None,
                penalty=algorithm_params["penalty"],
                Q=algorithm_params["Q"],
                learning_rate=0.01
            )
            subgraph_total_time = time.time() - subgraph_start_time

            # 记录子图级别结果
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

            # 5. 标准 QAOA 迭代优化
            optimized_coloring, opt_acc, conflict_counts, conflict_history, subgraph_opt_history = iterative_optimization_standard(
                graph=graph,
                subgraphs=subgraphs,
                sub_mappings=sub_mappings,
                subgraph_results=subgraph_results,
                max_k=algorithm_params["max_k"],
                p=algorithm_params["p"],
                num_steps=algorithm_params["num_steps"],
                max_iter=algorithm_params["max_iter"],
                adjacency_threshold=0.3,
                early_stop_threshold=algorithm_params["early_stop_threshold"],
                penalty=algorithm_params["penalty"],
                Q=algorithm_params["Q"],
                learning_rate=0.01,
                vertex_colors=None,
                nodes_to_recolor=None,
                dataset_name=dataset,
                graph_id=index
            )

            # 6. 统计结果
            final_coloring = optimized_coloring
            unique_colors = len(set(final_coloring.values())) if final_coloring else 0
            final_conflicts = count_conflicts(final_coloring, graph) if final_coloring else -1
            reoptimization_count = sum(
                1 for h in subgraph_opt_history
                if isinstance(h, tuple) and len(h) >= 4 and h[3] > 0
            )
            min_k_list = [r[0] for r in subgraph_results if r is not None and r[0] is not None]
            best_k_value = min(unique_colors, max(min_k_list) if min_k_list else unique_colors)

            print(f"\n===== Optimization Summary (Standard-QAOA Large) =====")
            print(f"Final Conflicts: {final_conflicts} (Total Edges: {num_edges})")
            print(f"Final Accuracy: {opt_acc:.4f}")
            print(f"Colors Used: {unique_colors} (Global max_k limit: {algorithm_params['max_k']})")
            print(f"Best k Value: {best_k_value}")

            # 经典算法对比输出
            if run_classical and classical_results:
                print(f"\n{'─'*50}")
                print(f"Classical Algorithms Comparison")
                print(f"{'─'*50}")

                graph_time = time.time() - graph_start_time
                qaoa_time_ms = graph_time * 1000
                qaoa_valid = "Yes" if final_conflicts == 0 else "No"

                print(f"{'Algorithm':<20} {'Colors':<10} {'Conflicts':<10} {'Time (ms)':<15} {'Valid'}")
                print(f"{'─'*65}")
                print(f"{'QAOA-Standard':<20} {unique_colors:<10} {final_conflicts:<10} {qaoa_time_ms:<15.2f} {qaoa_valid}")

                if 'greedy' in classical_results:
                    greedy = classical_results['greedy']
                    greedy_valid = "Yes" if greedy['conflicts'] == 0 else "No"
                    print(f"{'Greedy':<20} {greedy['num_colors']:<10} {greedy['conflicts']:<10} "
                          f"{greedy['execution_time_ms']:<15.2f} {greedy_valid}")

                if 'welch_powell' in classical_results:
                    wp = classical_results['welch_powell']
                    wp_valid = "Yes" if wp['conflicts'] == 0 else "No"
                    print(f"{'Welch-Powell':<20} {wp['num_colors']:<10} {wp['conflicts']:<10} "
                          f"{wp['execution_time_ms']:<15.2f} {wp_valid}")
                print(f"{'─'*65}")

            # 7. 子图着色可视化
            try:
                subgraph_colorings = [
                    get_subgraph_coloring(subgraph, final_coloring, mk)
                    for subgraph, mk in zip(subgraphs, min_k_list)
                ]
                plot_New_IDs_colored_subgraphs(
                    subgraphs, subgraph_colorings, sub_mappings, min_k_list,
                    title=f"{base_title} - Colored Subgraphs (Renumbered)",
                    filename=f"{base_title}_colored_subgraphs_renumbered",
                    output_dir=os.path.join(BASE_DIR, "large_subgraph_visualizations")
                )
                plot_Original_IDs_colored_subgraphs(
                    subgraphs, subgraph_colorings,
                    title=f"{base_title} - Colored Subgraphs (Original IDs)",
                    min_k_list=min_k_list, filename=f"{base_title}_colored_subgraphs_original",
                    output_dir=os.path.join(BASE_DIR, "large_subgraph_visualizations")
                )
            except Exception as e:
                print(f"Error in colored subgraph plotting for graph {index}: {str(e)}")
                traceback.print_exc()

            # 8. 最终图可视化
            graph_time = time.time() - graph_start_time
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

            # 9. 收集结果
            result = {
                "graph_index": index,
                "graph": graph,
                "final_coloring": final_coloring,
                "subgraphs": subgraphs,
                "sub_mappings": sub_mappings,
                "subgraph_results": subgraph_results,
                "sub_colorings": subgraph_colorings if 'subgraph_colorings' in locals() else [],
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

            # 10. 写全局日志
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
                    round(time.time() - total_start_time, 4)
                ])

        except Exception as e:
            print(f"Uncaught exception while processing graph {index}: {e}")
            traceback.print_exc()
            continue

    # 结果汇总
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"Large-scale Standard-QAOA all graphs processed, total time: {total_time:.1f}s")
    print(f"Successfully processed {len(all_results)}/{len(graphs)} graphs")
    print(f"CSV: {subgraph_csv} | Logs: {graph_log_csv}")
    print(f"{'='*60}")
    return all_results


def main_large_datasets():
    """
    主函数：处理大规模数据集 (cora, citeseer, pubmed)
    """
    args = parse_test_args()

    # 加载大规模数据集
    graphs = load_large_datasets()

    if not graphs:
        print("\n⚠️ 未能加载任何图数据，程序退出")
        return

    print(f"\n✓ 成功加载 {len(graphs)} 张大规模图")

    # 打印数据分析
    MAX_NODES_PER_SUBGRAPH = 50  # 根据硬件限制调整
    print_large_dataset_analysis(graphs, MAX_NODES_PER_SUBGRAPH)

    SEED = args.seed
    DATASET = "large_datasets"
    RUN_CLASSICAL = args.run_classical

    # 验证至少选择一种算法
    if not any([args.adapt, args.standard, args.adapt_noise]):
        print("⚠️ Must select at least one algorithm (--adapt/--standard/--adapt-noise)")
        print("Usage examples for large datasets:")
        print("  python Main_Multilevel_qaoa_large_graph.py --large-datasets --standard")
        print("  python Main_Multilevel_qaoa_large_graph.py --large-datasets --standard --run-classical")
        return

    results = {}

    # 运行 Adapt-QAOA
    if args.adapt:
        print("\n" + "="*60)
        print("Starting Adapt-QAOA on Large Datasets...")
        print("="*60)
        try:
            adapt_results = main_adapt_large(
                graphs=graphs,
                dataset=DATASET,
                graph_index=0,
                seed=SEED,
                run_classical=RUN_CLASSICAL
            )
            results['adapt'] = adapt_results
            print(f"\n✅ Adapt-QAOA completed, processed {len(adapt_results)} graphs")
        except Exception as e:
            print(f"\n⚠️ Adapt-QAOA failed: {e}")
            traceback.print_exc()

    # 运行 Standard-QAOA
    if args.standard:
        print("\n" + "="*60)
        print("Starting Standard-QAOA on Large Datasets...")
        print("="*60)
        try:
            standard_results = main_standard_large(
                graphs=graphs,
                dataset=DATASET,
                graph_index=0,
                seed=SEED,
                run_classical=RUN_CLASSICAL
            )
            results['standard'] = standard_results
            print(f"\n✅ Standard-QAOA completed, processed {len(standard_results)} graphs")
        except Exception as e:
            print(f"\n⚠️ Standard-QAOA failed: {e}")
            traceback.print_exc()

    # 运行 Noisy Adapt-QAOA
    if args.adapt_noise:
        print("\n" + "="*60)
        print(f"Starting Noisy Adapt-QAOA on Large Datasets (noise: {args.noise_prob})...")
        print("="*60)
        try:
            noise_results = main_adapt_noise_large(
                graphs=graphs,
                dataset=DATASET,
                graph_index=0,
                seed=SEED,
                run_classical=RUN_CLASSICAL,
                depolarizing_prob=args.noise_prob
            )
            results['adapt_noise'] = noise_results
            print(f"\n✅ Noisy Adapt-QAOA completed, processed {len(noise_results)} graphs")
        except Exception as e:
            print(f"\n⚠️ Noisy Adapt-QAOA failed: {e}")
            traceback.print_exc()

    # 输出结果汇总
    print("\n" + "="*70)
    print("                    Large Dataset Results Summary")
    print("="*70)

    for algo_name, algo_results in results.items():
        if not algo_results:
            continue

        print(f"\n【{algo_name.upper()}】")
        print("-" * 70)
        print(f"  Successfully Processed Graphs: {len(algo_results)}")
        print(f"  {'Graph Name':<20} {'Nodes':<8} {'Colors':<8} {'Conflicts':<10} {'Accuracy':<12} {'Time(s)':<10}")
        print("  " + "-" * 70)

        for r in algo_results:
            graph_name = os.path.splitext(r.get('base_title', 'unknown'))[0]
            nodes = r['num_nodes']
            colors = r['unique_colors']
            conflicts = r['final_conflicts']
            accuracy = r['accuracy']
            time_cost = r['processing_time']
            print(f"  {graph_name:<20} {nodes:<8} {colors:<8} {conflicts:<10} {accuracy:<12.4f} {time_cost:<10.2f}")

        # 计算统计
        avg_colors = sum(r['unique_colors'] for r in algo_results) / len(algo_results)
        avg_time = sum(r['processing_time'] for r in algo_results) / len(algo_results)
        avg_accuracy = sum(r['accuracy'] for r in algo_results) / len(algo_results)
        total_conflicts = sum(r['final_conflicts'] for r in algo_results)

        print("  " + "-" * 70)
        print(f"  Average Colors: {avg_colors:.2f}")
        print(f"  Average Time: {avg_time:.2f} s")
        print(f"  Average Accuracy: {avg_accuracy:.4f}")
        print(f"  Total Conflicts: {total_conflicts}")

    print("\n" + "="*70)
    print("✅ Large dataset experiments completed!")
    print("="*70 + "\n")



# ============================================================================
# 说明：本文件已被简化，主函数已移除
# ============================================================================
#
# 本文件包含大规模数据集处理函数，但主函数入口已移至 Main_Multilevel_qaoa.py
# 如需运行大规模数据集实验，请使用 run_experiments.py 或参考以下命令：
#
# Adapt-QAOA 大规模数据集
# python Main_Multilevel_qaoa_large_graph.py --large-datasets --adapt
#
# Standard-QAOA 大规模数据集
# python Main_Multilevel_qaoa_large_graph.py --large-datasets --standard
#
# Noisy Adapt-QAOA 大规模数据集
# python Main_Multilevel_qaoa_large_graph.py --large-datasets --adapt-noise --noise-prob 0.05
#
# 带经典算法对比
# python Main_Multilevel_qaoa_large_graph.py --large-datasets --standard --run-classical
#
# ============================================================================




'''