# -*- coding: utf-8 -*-
"""
大规模图数据集的经典算法着色 (Greedy 和 Tabu)
参考 Main_Multilevel_qaoa_large_graph.py 读取大规模数据
数据集: citeseer.col, pubmed.col
"""
import os
import sys
import time
import csv
import traceback
import networkx as nx

# 添加模块路径
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_FILE_DIR)  # HadaQAOA
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, "classical_algorithms"))
sys.path.insert(0, os.path.join(PARENT_DIR, "standard_and_adapt_QAOA"))

from greedy import GreedyColoring
from unified_data_processor import UnifiedDataProcessor
from classical_tabu_coloring import process_single_graph as process_tabu
from graph_loader import read_col_file


def load_large_datasets():
    """
    加载大规模数据集
    参考 Main_Multilevel_qaoa_large_graph.py 中的 load_large_datasets()
    """
    # 数据目录: HAdaQAOA/Data/Large_datesets/
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "Data", "Large_datesets")

    large_datasets = [
        # "cora.col",
        # "citeseer.col",
        "pubmed.col"
    ]

    print(f"\n{'='*70}")
    print(f"加载大规模图数据...")
    print(f"{'='*70}")
    print(f"  数据目录: {DATA_DIR}")

    graphs = []
    for dataset_name in large_datasets:
        dataset_path = os.path.join(DATA_DIR, dataset_name)
        if os.path.exists(dataset_path):
            print(f"\n  正在加载: {dataset_name}")
            try:
                graph = read_col_file(dataset_path)
                if graph is not None:
                    graph.file_name = dataset_name
                    graphs.append(graph)
                    print(f"    ✓ 成功加载 {dataset_name}")
                    print(f"      节点数: {graph.number_of_nodes()}")
                    print(f"      边数: {graph.number_of_edges()}")
                    print(f"      平均度数: {2 * graph.number_of_edges() / graph.number_of_nodes():.2f}")
                else:
                    print(f"    ✗ 加载失败: {dataset_name}")
            except Exception as e:
                print(f"    ✗ 加载 {dataset_name} 时出错: {e}")
                traceback.print_exc()
        else:
            print(f"  ✗ 文件不存在: {dataset_path}")

    print(f"\n{'='*70}")
    print(f"✓ 总共加载了 {len(graphs)} 个大规模图")
    print(f"{'='*70}\n")

    return graphs


def run_greedy_coloring(graph, dataset_name, graph_index):
    """
    运行贪心算法进行图着色
    """
    print(f"\n{'─'*70}")
    print(f"[Greedy] 正在处理图 {graph_index}: {dataset_name}")
    print(f"{'─'*70}")

    start_time = time.perf_counter()
    try:
        greedy = GreedyColoring(graph)
        coloring, num_colors = greedy.execute()
        conflicts = count_conflicts(coloring, graph)
        exec_time = (time.perf_counter() - start_time) * 1000  # 毫秒

        result = {
            'algorithm': 'Greedy',
            'dataset': dataset_name,
            'graph_index': graph_index,
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'num_colors': num_colors,
            'conflicts': conflicts,
            'execution_time_ms': round(exec_time, 4),
            'is_valid': conflicts == 0,
            'coloring': coloring
        }

        print(f"  颜色数: {num_colors}")
        print(f"  冲突数: {conflicts}")
        print(f"  是否有效: {'✓ 有效' if conflicts == 0 else '✗ 无效'}")
        print(f"  执行时间: {exec_time:.4f} ms")

        return result
    except ValueError as e:
        # 处理greedy.py抛出的验证异常
        if "Invalid coloring" in str(e):
            print(f"  ⚠️ Greedy算法验证失败，使用内部着色结果重新计算...")
            try:
                # 尝试获取内部着色结果
                greedy = GreedyColoring(graph)
                # 手动执行着色逻辑，绕过验证
                for node in sorted(greedy.nodes, key=lambda x: graph.degree(x), reverse=True):
                    neighbor_colors = {greedy.coloring[neigh] for neigh in graph.neighbors(node) if neigh in greedy.coloring}
                    color = 0
                    while color in neighbor_colors:
                        color += 1
                    greedy.coloring[node] = color
                    greedy.used_colors.add(color)

                coloring = greedy.coloring
                num_colors = len(greedy.used_colors)
                conflicts = count_conflicts(coloring, graph)
                exec_time = (time.perf_counter() - start_time) * 1000

                result = {
                    'algorithm': 'Greedy',
                    'dataset': dataset_name,
                    'graph_index': graph_index,
                    'num_nodes': graph.number_of_nodes(),
                    'num_edges': graph.number_of_edges(),
                    'num_colors': num_colors,
                    'conflicts': conflicts,
                    'execution_time_ms': round(exec_time, 4),
                    'is_valid': conflicts == 0,
                    'coloring': coloring,
                    'validation_failed': True
                }

                print(f"  颜色数: {num_colors}")
                print(f"  冲突数: {conflicts}")
                print(f"  是否有效: {'✓ 有效' if conflicts == 0 else '✗ 无效'}")
                print(f"  执行时间: {exec_time:.4f} ms")
                print(f"  ⚠️ 警告: 算法内部验证失败，但已生成着色方案")

                return result
            except Exception as inner_e:
                print(f"  ❌ 重新尝试也失败: {inner_e}")
                traceback.print_exc()
                return None
        else:
            print(f"  ❌ 贪心算法执行失败: {e}")
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"  ❌ 贪心算法执行失败: {e}")
        traceback.print_exc()
        return None


def run_tabu_coloring(graph, dataset_name, graph_index, data_processor):
    """
    运行Tabu搜索算法进行图着色
    """
    print(f"\n{'─'*70}")
    print(f"[Tabu] 正在处理图 {graph_index}: {dataset_name}")
    print(f"{'─'*70}")

    start_time = time.perf_counter()
    try:
        # 创建临时col文件
        temp_col_file = f"temp_{dataset_name.replace('.col', '')}_{graph_index}.col"
        with open(temp_col_file, "w") as f:
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            f.write(f"p edge {n_nodes} {n_edges}\n")
            for u, v in graph.edges:
                f.write(f"e {u} {v}\n")

        # 解析图并设置合理的色数上限
        parsed_graph = data_processor.parse_col_file(temp_col_file)
        max_iterations = 50000  # 大规模图增加迭代次数
        max_retries = 3

        # 运行Tabu算法
        tabu_res = process_tabu(
            temp_col_file,
            parsed_graph,
            max_iterations_base=max_iterations,
            max_retries=max_retries
        )

        # 清理临时文件
        if os.path.exists(temp_col_file):
            os.remove(temp_col_file)

        exec_time = tabu_res["execution_time_ms"]
        result = {
            'algorithm': 'TabuCol',
            'dataset': dataset_name,
            'graph_index': graph_index,
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'num_colors': tabu_res["num_colors"],
            'conflicts': tabu_res["conflicts"],
            'execution_time_ms': round(exec_time, 4),
            'is_valid': tabu_res["conflicts"] == 0,
            'coloring': tabu_res["coloring"]
        }

        print(f"  颜色数: {tabu_res['num_colors']}")
        print(f"  冲突数: {tabu_res['conflicts']}")
        print(f"  是否有效: {'✓ 有效' if tabu_res['conflicts'] == 0 else '✗ 无效'}")
        print(f"  执行时间: {exec_time:.4f} ms")

        return result
    except Exception as e:
        print(f"  ❌ Tabu算法执行失败: {e}")
        traceback.print_exc()
        return None


def count_conflicts(coloring, graph):
    """
    计算着色方案中的冲突数
    """
    conflicts = 0
    for u, v in graph.edges():
        if u in coloring and v in coloring and coloring[u] == coloring[v]:
            conflicts += 1
    return conflicts


def save_results_to_csv(all_results, output_dir="csvs"):
    """
    将结果保存到CSV文件
    """
    os.makedirs(output_dir, exist_ok=True)

    # 主结果CSV
    csv_path = os.path.join(output_dir, "large_datasets_classical_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "dataset", "graph_index",
            "num_nodes", "num_edges", "num_colors",
            "conflicts", "is_valid", "execution_time_ms",
            "validation_failed"
        ])

        for result in all_results:
            if result is None:
                continue
            writer.writerow([
                result['algorithm'],
                result['dataset'],
                result['graph_index'],
                result['num_nodes'],
                result['num_edges'],
                result['num_colors'],
                result['conflicts'],
                result['is_valid'],
                result['execution_time_ms'],
                result.get('validation_failed', False)
            ])

    print(f"\n✓ 结果已保存到: {csv_path}")
    return csv_path


def print_summary(all_results):
    """
    打印结果摘要
    """
    print(f"\n{'='*70}")
    print(f"                      实验结果摘要")
    print(f"{'='*70}\n")

    # 按算法分组
    algorithm_results = {}
    for result in all_results:
        if result is None:
            continue
        algo = result['algorithm']
        if algo not in algorithm_results:
            algorithm_results[algo] = []
        algorithm_results[algo].append(result)

    # 打印每个算法的结果
    for algo, results in algorithm_results.items():
        print(f"【{algo}】")
        print("-" * 70)
        print(f"  {'图名称':<20} {'节点':<10} {'边':<10} {'颜色':<10} {'冲突':<10} {'时间(ms)':<15}")
        print("  " + "-" * 70)

        for r in results:
            dataset_name = r['dataset']
            print(f"  {dataset_name:<20} {r['num_nodes']:<10} {r['num_edges']:<10} "
                  f"{r['num_colors']:<10} {r['conflicts']:<10} {r['execution_time_ms']:<15.4f}")

        # 计算统计信息
        avg_colors = sum(r['num_colors'] for r in results) / len(results)
        avg_time = sum(r['execution_time_ms'] for r in results) / len(results)
        total_conflicts = sum(r['conflicts'] for r in results)
        valid_count = sum(1 for r in results if r['is_valid'])

        print("  " + "-" * 70)
        print(f"  平均颜色数: {avg_colors:.2f}")
        print(f"  平均时间: {avg_time:.4f} ms")
        print(f"  总冲突数: {total_conflicts}")
        print(f"  有效解数量: {valid_count}/{len(results)}")
        print()

    # 算法对比
    if len(algorithm_results) > 1:
        print(f"\n{'='*70}")
        print(f"                      算法对比")
        print(f"{'='*70}\n")
        print(f"  {'算法':<15} {'平均颜色数':<15} {'平均时间(ms)':<20} {'有效解率':<15}")
        print("  " + "-" * 65)

        for algo, results in algorithm_results.items():
            avg_colors = sum(r['num_colors'] for r in results) / len(results)
            avg_time = sum(r['execution_time_ms'] for r in results) / len(results)
            valid_rate = sum(1 for r in results if r['is_valid']) / len(results) * 100
            print(f"  {algo:<15} {avg_colors:<15.2f} {avg_time:<20.4f} {valid_rate:<14.1f}%")

    print(f"\n{'='*70}")


def main():
    """
    主函数：运行大规模图数据的经典算法着色
    """
    print(f"\n{'='*70}")
    print(f"         大规模图数据集经典算法着色实验")
    print(f"         Large Graph Classical Coloring Experiments")
    print(f"{'='*70}")

    # 初始化数据处理器
    data_processor = UnifiedDataProcessor()

    # 加载大规模数据集
    graphs = load_large_datasets()

    if not graphs:
        print("\n❌ 没有加载到图数据，程序退出")
        print("   提示: 请确保数据文件存在于 HAdaQAOA/Data/Large_datesets/ 目录")
        return

    # 运行算法
    all_results = []

    print(f"\n{'='*70}")
    print(f"开始运行经典算法...")
    print(f"{'='*70}")

    for idx, graph in enumerate(graphs, start=1):
        dataset_name = getattr(graph, 'file_name', f'graph_{idx}')

        # 运行Greedy算法
        greedy_result = run_greedy_coloring(graph, dataset_name, idx)
        if greedy_result:
            all_results.append(greedy_result)

        # 运行Tabu算法
        tabu_result = run_tabu_coloring(graph, dataset_name, idx, data_processor)
        if tabu_result:
            all_results.append(tabu_result)

    # 保存结果
    csv_path = save_results_to_csv(all_results)

    # 打印摘要
    print_summary(all_results)

    print(f"\n✅ 所有实验完成!")
    print(f"   结果已保存到: {csv_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
