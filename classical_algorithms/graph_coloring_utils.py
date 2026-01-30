"""
图着色算法通用工具模块
提供数据读取、结果保存、可视化等通用功能
"""

import os
import csv
import time
import networkx as nx
import matplotlib
# 使用非交互式后端，图片显示后不阻塞程序继续执行
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from greedy import GraphColoringVisualizer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class GraphColoringUtils:
    """图着色算法通用工具类"""
    
    def __init__(self, data_dir="../../Data/instances", results_dir="coloring_results"):
        """
        初始化工具类
        Args:
            data_dir: 数据文件目录，默认相对路径
            results_dir: 结果保存目录，默认相对路径
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        # 注释掉结果目录的创建
        # os.makedirs(self.results_dir, exist_ok=True)
        # print(f"📁 结果目录已准备：{os.path.abspath(self.results_dir)}")
        pass
    
    def parse_col_file(self, file_path):
        """
        读取 DIMACS COLOR 格式（.col）文件，返回 networkx.Graph
        支持以 'c' 开头的注释行、'p' 描述行、'e' 边行
        
        Args:
            file_path: .col 文件路径
            
        Returns:
            networkx.Graph: 解析后的图对象
        """
        graph = nx.Graph()
        print(f"📂 解析文件: {os.path.basename(file_path)} (路径: {file_path})")
        
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("c"):
                        continue
                    if line.startswith("p"):
                        parts = line.split()
                        if len(parts) < 4 or parts[1] != "edge":
                            print(f"⚠️ 跳过格式错误的 'p' 行（第{line_no}行）：{line}")
                            continue
                        try:
                            n_nodes = int(parts[2])
                            n_edges = int(parts[3])
                            graph.add_nodes_from(range(1, n_nodes + 1))
                            print(f"  成功添加 {n_nodes} 个节点（预期边数：{n_edges}）")
                        except ValueError as e:
                            print(f"❌ 'p' 行解析错误（第{line_no}行）：{e}，内容：{parts}")
                            continue
                    if line.startswith("e"):
                        parts = line.split()
                        if len(parts) < 3:
                            print(f"⚠️ 跳过格式错误的 'e' 行（第{line_no}行）：{line}")
                            continue
                        try:
                            u, v = map(int, parts[1:3])
                            max_node = graph.number_of_nodes()
                            if u < 1 or v < 1 or u > max_node or v > max_node:
                                print(f"⚠️ 边 ({u},{v}) 包含无效节点（节点范围1-{max_node}），跳过")
                                continue
                            graph.add_edge(u, v)
                        except ValueError as e:
                            continue
            print(f"📊 解析完成：节点数={graph.number_of_nodes()}, 实际边数={graph.number_of_edges()}")
            if graph.number_of_nodes() == 0:
                print("⚠️ 解析结果为空白图（无节点）")
            return graph
        except Exception as e:
            print(f"❌ 解析文件时发生错误：{str(e)}")
            return None
    
    def get_col_files(self):
        """
        获取数据目录下的所有 .col 文件
        
        Returns:
            list: .col 文件路径列表
        """
        if not os.path.exists(self.data_dir):
            print(f"❌ 数据目录不存在：{self.data_dir}")
            return []
        
        import glob
        col_files = glob.glob(os.path.join(self.data_dir, "*.col"))
        
        if not col_files:
            print(f"⚠️ 未找到任何 .col 文件（路径：{self.data_dir}），请检查路径是否正确")
        else:
            print(f"✅ 共找到 {len(col_files)} 个 .col 文件")
            
        return col_files
    
    def calculate_conflicts(self, graph, coloring):
        """
        计算着色方案中的冲突数
        
        Args:
            graph: networkx.Graph 图对象
            coloring: dict 着色方案 {节点: 颜色}
            
        Returns:
            int: 冲突数量
        """
        if not graph or not coloring:
            return 0
        
        conflicts = 0
        for u, v in graph.edges():
            if u in coloring and v in coloring:
                if coloring[u] == coloring[v]:
                    conflicts += 1
        return conflicts
    
    def normalize_coloring(self, coloring):
        """
        将颜色值归一化到 0~num_colors-1 范围，避免索引越界
        
        Args:
            coloring: dict 原始着色方案
            
        Returns:
            tuple: (归一化后的着色方案, 实际颜色数)
        """
        if not coloring:
            return {}, 0
        
        # 获取所有唯一颜色值并排序
        color_values = sorted(set(coloring.values()))
        # 建立颜色值到连续索引的映射
        color_mapping = {v: i for i, v in enumerate(color_values)}
        # 重新映射着色方案
        normalized_coloring = {node: color_mapping[color] for node, color in coloring.items()}
        return normalized_coloring, len(color_values)
    
    def save_results_to_csv(self, results, csv_filename):
        """
        将结果保存为CSV文件
        
        Args:
            results: list 结果字典列表
            csv_filename: str CSV文件名（不含路径）
        """
        if not results:
            print("📌 save_results_to_csv: 输入结果列表为空，无需保存")
            return
        
        # 完整文件路径
        output_file = os.path.join(self.results_dir, csv_filename)
        
        # 定义CSV文件的列名
        fieldnames = [
            'filename', 'num_nodes', 'num_edges', 
            'num_colors', 'conflicts', 'execution_time_ms', 
            'is_valid', 'algorithm'
        ]
        
        # 检查是否已存在文件，避免重复写入
        existed = set()
        if os.path.isfile(output_file):
            try:
                with open(output_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    existed = {row['filename'] for row in reader}
            except Exception as e:
                print(f"⚠️ 读取现有CSV文件时出错：{e}")
        
        # 过滤掉已存在的记录
        new_results = [r for r in results if r.get('filename') not in existed]
        
        if not new_results:
            print("ℹ️ 所有数据均已存在，无需追加")
            return
        
        try:
            # 写入新数据
            with open(output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # 如果文件不存在或为空，写入表头
                if not existed:
                    writer.writeheader()
                
                # 过滤并写入数据
                for result in new_results:
                    filtered_result = {k: v for k, v in result.items() if k in fieldnames}
                    writer.writerow(filtered_result)
            
            print(f"✅ 已追加 {len(new_results)} 条新记录到 {output_file}")
            
        except Exception as e:
            print(f"❌ 保存CSV文件时出错：{e}")
    
    def visualize_coloring(self, graph, coloring, filename, num_colors, exec_time, 
                          algorithm_name="Algorithm"):
        """
        可视化图着色结果并保存为PDF
        
        Args:
            graph: networkx.Graph 图对象
            coloring: dict 着色方案
            filename: str 原始文件名
            num_colors: int 使用的颜色数
            exec_time: float 执行时间（毫秒）
            algorithm_name: str 算法名称
            
        Returns:
            str: 保存的PDF文件路径
        """
        if not coloring or num_colors <= 0:
            print(f"⚠️ 着色方案为空，跳过可视化：{filename}")
            return None
        
        try:
            # 先归一化颜色值，确保颜色值是连续的
            normalized_coloring, actual_colors = self.normalize_coloring(coloring)
            
            # 生成与实际颜色数匹配的颜色列表
            if actual_colors <= 20:
                color_map = plt.colormaps.get('tab20')  # 颜色少的时候用tab20，区分度高
            else:
                color_map = plt.colormaps.get('hsv')    # 颜色多的时候用hsv
            colors = [color_map(i / max(actual_colors, 1)) for i in range(actual_colors)]
            
            # 初始化可视化器，使用归一化后的着色方案
            vis = GraphColoringVisualizer(
                graph=graph,
                coloring=normalized_coloring,
                filename=filename,
                num_colors=actual_colors,
                exec_time=exec_time
            )
            vis.colors = colors
            
            # 生成保存文件名
            base_filename = os.path.splitext(filename)[0]
            save_filename = f"{base_filename}_{algorithm_name.lower()}_coloring.pdf"
            save_path = os.path.join(self.results_dir, save_filename)
            
            # 保存可视化结果
            vis.save_combined_visualization(save_path)
            print(f"  📊 可视化结果已保存至：{save_path}")
            
            return save_path
            
        except Exception as e:
            print(f"⚠️ 可视化时出错：{str(e)}")
            return None
    
    def plot_original_graph(self, graph, title="原始图可视化", save_filename=None):
        """
        可视化原始图（无着色）
        
        Args:
            graph: networkx.Graph 图对象
            title: str 图标题
            save_filename: str 保存文件名（可选）
            
        Returns:
            str: 保存的文件路径（如果保存）
        """
        if not graph or len(graph.nodes) == 0:
            print("警告：无效或空的图，无法可视化")
            return None
        
        num_nodes = len(graph.nodes)
        
        # 动态画布大小
        fig_w = min(12 + (num_nodes // 8) * 2, 24)
        fig_h = min(10 + (num_nodes // 8) * 1.6, 20)
        plt.figure(figsize=(fig_w, fig_h))
        
        # 自动计算布局参数
        k = 2.0 / np.sqrt(num_nodes)
        
        pos = nx.spring_layout(
            graph,
            seed=42,
            scale=1.2,
            k=k,
            iterations=200
        )
        
        # 绘制图
        nx.draw_networkx_edges(graph, pos, width=1.8, alpha=0.7, edge_color='#888888')
        nx.draw_networkx_nodes(graph, pos,
                               node_color='#AAAAAA',
                               node_size=300,
                               edgecolors='#333333',
                               linewidths=1.5)
        nx.draw_networkx_labels(graph, pos,
                                labels={n: str(n) for n in graph.nodes()},
                                font_size=10,
                                font_family='sans-serif',
                                font_weight='bold')
        
        isolated = sum(1 for n in graph.nodes if graph.degree(n) == 0)
        plt.title(f"{title}\n(Nodes={num_nodes}, Isolated={isolated}, Edges={graph.number_of_edges()})",
                  fontsize=16, pad=25)
        plt.axis('off')
        plt.tight_layout(pad=2.0)
        
        if save_filename:
            save_path = os.path.join(self.results_dir, save_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 原始图已保存至：{save_path}")
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def process_graphs_batch(self, algorithm_func, algorithm_name, **algorithm_params):
        """
        批量处理图的通用函数
        
        Args:
            algorithm_func: function 算法处理函数，接受(filename, graph, **kwargs)参数
            algorithm_name: str 算法名称
            **algorithm_params: 传递给算法的额外参数
            
        Returns:
            list: 所有图的处理结果
        """
        col_files = self.get_col_files()
        if not col_files:
            return []
        
        print(f"🚀 开始批量处理 {len(col_files)} 个图，使用算法：{algorithm_name}")
        all_results = []
        
        for idx, file_path in enumerate(col_files, 1):
            filename = os.path.basename(file_path)
            print(f"\n===== 处理第 {idx}/{len(col_files)} 张图：{filename} =====")
            
            try:
                if not os.path.isfile(file_path):
                    print(f"❌ 文件不存在：{file_path}")
                    continue
                
                # 解析图
                graph = self.parse_col_file(file_path)
                if graph is None:
                    print(f"⚠️ 跳过无效图：{filename}")
                    continue
                
                # 调用算法处理
                start_time = time.perf_counter()
                result = algorithm_func(filename, graph, **algorithm_params)
                exec_time = (time.perf_counter() - start_time) * 1000
                
                # 标准化结果格式
                if result:
                    result.update({
                        'algorithm': algorithm_name,
                        'execution_time_ms': round(exec_time, 2)
                    })
                    all_results.append(result)
                    
                    # 打印结果摘要
                    print(f"📊 结果摘要：{filename} | 节点数：{result['num_nodes']} | "
                          f"颜色数：{result['num_colors']} | 冲突数：{result['conflicts']} | "
                          f"耗时：{result['execution_time_ms']}ms | 有效：{result['is_valid']}")
                    
                    # 自动保存可视化（如果算法没有生成）
                    if result.get('coloring') and not os.path.exists(
                        os.path.join(self.results_dir, f"{os.path.splitext(filename)[0]}_{algorithm_name.lower()}_coloring.pdf")
                    ):
                        self.visualize_coloring(
                            graph, result['coloring'], filename, 
                            result['num_colors'], result['execution_time_ms'], 
                            algorithm_name
                        )
                else:
                    print(f"⚠️ 未生成结果：{filename}")
                    
            except Exception as e:
                print(f"❌ 处理 {filename} 时出错：{str(e)}")
                import traceback
                traceback.print_exc()
        
        # 自动保存CSV结果
        csv_filename = f"{algorithm_name.lower()}_coloring_results.csv"
        self.save_results_to_csv(all_results, csv_filename)
        
        print(f"\n🎉 批量处理完成，共处理 {len(all_results)} 个图")
        print(f"📊 结果已保存至：{os.path.join(self.results_dir, csv_filename)}")
        
        return all_results


# 全局工具实例，方便直接使用
utils = GraphColoringUtils()