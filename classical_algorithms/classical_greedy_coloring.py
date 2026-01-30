"""
经典贪心算法图着色
使用通用工具类处理数据读取、结果保存和可视化
支持从 graph_loader 加载 .col 和 .pkl 格式数据
"""

import os
import sys
import time
from greedy import process_single_graph, GreedyColoring, GraphColoringVisualizer
from graph_coloring_utils import GraphColoringUtils

# 添加 graph_loader 所在的路径
graph_loader_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'standard_and_adapt_QAOA', 'graph_loader.py')
graph_loader_dir = os.path.dirname(graph_loader_path)
if graph_loader_dir not in sys.path:
    sys.path.insert(0, graph_loader_dir)

from graph_loader import load_graphs_from_dir

# 初始化工具类
utils = GraphColoringUtils(
    data_dir=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'Data', 'instances'
    ),
    results_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coloring_results')
)


def greedy_algorithm_handler(filename, graph, visualize=True):
    """
    Greedy算法处理函数的包装器
    Args:
        filename: str 文件名
        graph: networkx.Graph 图对象
        visualize: bool 是否可视化
    Returns:
        dict: 处理结果
    """
    # 调用原始的process_single_graph函数
    result = process_single_graph(filename, graph, visualize=visualize)
    
    if result:
        # 标准化结果格式，确保包含必需的字段
        result.setdefault('filename', filename)
        result.setdefault('num_nodes', graph.number_of_nodes())
        result.setdefault('num_edges', graph.number_of_edges())
        result.setdefault('is_valid', result.get('conflicts', 0) == 0)
        
        return result
    return None


def greedy_coloring_with_loader(graph, filename, save_dir='./coloring_results'):
    """
    使用贪心算法对图进行着色，并保存可视化结果
    
    Args:
        graph: networkx.Graph 图对象
        filename: str 文件名
        save_dir: str 结果保存目录
    
    Returns:
        dict: 处理结果字典
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录执行时间
    start_time = time.perf_counter()
    
    # 执行贪心着色
    greedy = GreedyColoring(graph)
    coloring, num_colors = greedy.execute()
    
    # 计算执行时间（转换为毫秒）
    exec_time = (time.perf_counter() - start_time) * 1000
    
    # 验证着色有效性
    conflicts = sum(1 for u, v in graph.edges() if coloring[u] == coloring[v])
    is_valid = conflicts == 0
    
    # 生成组合可视化并保存为PDF
    base_name = os.path.splitext(filename)[0]
    combined_save_path = os.path.join(save_dir, f"{base_name}_greedy_combined.pdf")
    visualizer = GraphColoringVisualizer(
        graph=graph,
        coloring=coloring,
        filename=filename,
        num_colors=num_colors,
        exec_time=exec_time
    )
    visualizer.save_combined_visualization(combined_save_path)
    
    print(f"✅ 贪心着色完成: {filename}")
    print(f"   节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
    print(f"   使用颜色数: {num_colors}, 冲突数: {conflicts}")
    print(f"   执行时间: {exec_time:.2f}ms")
    print(f"   可视化已保存至: {combined_save_path}")
    
    return {
        'filename': filename,
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'num_colors': num_colors,
        'conflicts': conflicts,
        'execution_time_ms': round(exec_time, 2),
        'is_valid': is_valid,
        'coloring': coloring,
        'save_path': save_dir
    }


if __name__ == "__main__":
    # 获取数据目录路径
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'Data', 'instances'
    )
    
    print("=" * 60)
    print("经典贪心算法图着色")
    print("=" * 60)
    print(f"数据目录: {data_dir}")
    print("=" * 60)
    
    # 使用 graph_loader 加载COL数据
    print("\n📂 正在加载图数据...")
    graphs = load_graphs_from_dir(data_dir, format_type='col')
    
    if not graphs:
        print("❌ 未能加载任何图数据，请检查数据目录")
        sys.exit(1)
    
    print(f"\n✅ 成功加载 {len(graphs)} 张图\n")
    
    # 批量处理所有图
    all_results = []
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coloring_results')
    
    for idx, graph in enumerate(graphs, 1):
        filename = getattr(graph, 'file_name', f'graph_{idx}')
        print(f"\n{'=' * 60}")
        print(f"处理第 {idx}/{len(graphs)} 张图: {filename}")
        print(f"{'=' * 60}")
        
        try:
            result = greedy_coloring_with_loader(graph, filename, save_dir)
            all_results.append(result)
        except Exception as e:
            print(f"❌ 处理 {filename} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 保存CSV结果
    if all_results:
        csv_filename = os.path.join(save_dir, 'greedy_coloring_results.csv')
        utils.save_results_to_csv(all_results, os.path.basename(csv_filename))
    
    print("\n" + "=" * 60)
    print("🎉 所有图处理完成！")
    print("=" * 60)
    print(f"总共处理: {len(all_results)} 张图")
    print(f"结果保存目录: {save_dir}")
    
    # 打印统计信息
    if all_results:
        total_nodes = sum(r['num_nodes'] for r in all_results)
        total_edges = sum(r['num_edges'] for r in all_results)
        total_colors = sum(r['num_colors'] for r in all_results)
        avg_colors = total_colors / len(all_results)
        total_time = sum(r['execution_time_ms'] for r in all_results)
        
        print(f"\n📊 统计信息:")
        print(f"   总节点数: {total_nodes}")
        print(f"   总边数: {total_edges}")
        print(f"   平均使用颜色数: {avg_colors:.2f}")
        print(f"   总执行时间: {total_time:.2f}ms")
        print(f"   平均执行时间: {total_time / len(all_results):.2f}ms")