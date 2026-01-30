from collections import Counter
import time

import networkx as nx
import numpy as np
import os
import pickle
import matplotlib
# 使用非交互式后端，图片显示后不阻塞程序继续执行
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


# -------------------------- Greedy Coloring Algorithm (with visualization integration) --------------------------
class GreedyColoring:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(graph.nodes())
        # Sort nodes by descending degree (core greedy strategy)
        self.node_order = sorted(self.nodes, key=lambda x: graph.degree(x), reverse=True)
        self.coloring = {}  # {node: color}
        self.used_colors = set()

    def execute(self):
        """Execute greedy coloring, return coloring scheme and number of colors"""
        for node in self.node_order:
            # Collect colors used by neighbors
            neighbor_colors = {self.coloring[neigh] for neigh in self.graph.neighbors(node) if neigh in self.coloring}
            # Find the smallest available color
            color = 0
            while color in neighbor_colors:
                color += 1
            # Assign color
            self.coloring[node] = color
            self.used_colors.add(color)
        # Verify validity (ensure no conflicts)
        if not self._is_valid():
            raise ValueError("Invalid coloring result: adjacent nodes have the same color")
        return self.coloring, len(self.used_colors)

    def _is_valid(self):
        """Verify coloring validity (adjacent nodes have different colors)"""
        for u, v in self.graph.edges():
            if self.coloring[u] == self.coloring[v]:
                return False
        return True


def generate_distinct_colors(num_colors):
    """Generate a set of visually distinct colors"""
    if num_colors == 0:
        return []
    # Use HSV color space with evenly distributed hues for distinctness
    hues = np.linspace(0, 1, num_colors, endpoint=False)
    saturation = 0.8 if num_colors < 5 else 0.6 if num_colors < 10 else 0.5
    value = 0.9  # Fixed lightness for bright colors
    colors = [hsv_to_rgb((h, saturation, value)) for h in hues]
    return colors


def calculate_color_statistics(coloring):
    """Statistics on color usage (for auxiliary charts)"""
    color_count = Counter(coloring.values())
    # Sort by color ID to ensure consistent chart order
    sorted_colors = sorted(color_count.items(), key=lambda x: x[0])
    color_ids = [str(item[0]) for item in sorted_colors]  # Color IDs (as strings for display)
    node_counts = [item[1] for item in sorted_colors]  # Number of nodes for each color
    return color_ids, node_counts


def visualize_coloring(graph, coloring, title, save_path=None, figsize=(10, 8)):
    """Visualize graph coloring results and save as PDF if save_path is provided"""
    # Generate color list matching the number of colors
    num_colors = len(set(coloring.values()))
    colors = generate_distinct_colors(num_colors)

    # Assign corresponding colors to each node
    node_colors = [colors[coloring[node]] for node in graph.nodes()]

    # Draw the graph
    plt.figure(figsize=figsize)
    # Use spring layout for aesthetic and effective structure display
    pos = nx.spring_layout(graph, seed=42)  # Fixed seed for consistent layout

    # Draw nodes and edges
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=300, edgecolors='black')
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=10)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    # Save as PDF if path is provided, otherwise show
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()  # Close figure to free memory
    else:
        plt.show()


# -------------------------- Multi-dimensional Visualization Class --------------------------
class GraphColoringVisualizer:
    """Graph coloring visualization class, supporting structure display, color statistics, and performance labeling"""

    def __init__(self, graph, coloring, filename, num_colors, exec_time):
        self.graph = graph  # NetworkX graph object
        self.coloring = coloring  # Coloring scheme ({node: color})
        self.filename = filename  # Graph filename (for title annotation)
        self.num_colors = num_colors  # Total number of colors used
        self.exec_time = exec_time  # Algorithm execution time (ms)
        self.colors = generate_distinct_colors(num_colors)  # Color scheme

    def plot_color_distribution(self, ax):
        """Plot color usage distribution bar chart (to analyze allocation rationality)"""
        color_ids, node_counts = calculate_color_statistics(self.coloring)

        # Fill bars with corresponding colors for better association
        bars = ax.bar(color_ids, node_counts, color=self.colors, edgecolor='black', linewidth=0.8)

        # Label node counts on top of bars
        for bar, count in zip(bars, node_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    str(count), ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Color ID', fontsize=10)
        ax.set_ylabel('Number of Nodes', fontsize=10)
        ax.set_title('Color Usage Distribution', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)  # Only horizontal grid lines to avoid clutter

    def plot_colored_graph(self, ax):
        """Plot colored graph structure (optimized layout and node display)"""
        # Fixed layout seed for consistent positioning
        pos = nx.spring_layout(self.graph, seed=42, k=2.5)  # k controls node spacing

        # Assign corresponding colors to each node
        node_colors = [self.colors[self.coloring[node]] for node in self.graph.nodes()]

        # Draw nodes (with black borders for better definition)
        nx.draw_networkx_nodes(
            self.graph, pos, ax=ax,
            node_color=node_colors, node_size=400,
            edgecolors='black', linewidths=1.2
        )

        # Draw edges (semi-transparent to avoid obscuring nodes)
        nx.draw_networkx_edges(
            self.graph, pos, ax=ax,
            alpha=0.6, width=1.0, edge_color='#666666'
        )

        # Draw node labels (ensure readability)
        nx.draw_networkx_labels(
            self.graph, pos, ax=ax,
            font_size=9, font_weight='bold', font_family='monospace'
        )

        ax.set_title(
            f'Graph Coloring Result\n{self.filename}\nColors: {self.num_colors} | Time: {self.exec_time:.2f}ms',
            fontsize=11, fontweight='bold', pad=20
        )
        ax.axis('off')  # Hide coordinate axes

    def save_combined_visualization(self, save_path, figsize=(14, 6)):
        """Save combined visualization as PDF (graph structure on left, color stats on right)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Draw colored graph and statistics chart
        self.plot_colored_graph(ax1)
        self.plot_color_distribution(ax2)

        # Adjust subplot spacing to avoid overlap
        plt.tight_layout(pad=3.0)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save as PDF
        plt.title(f"Coloring Result for {self.filename}", fontsize=12)
        # 保存逻辑...
        plt.savefig(save_path, format='pdf')
        plt.close()

# Usage example with dataset integration
def solve_with_greedy(graph, save_path=None, title_suffix=""):
    """Wrapper function for solving graph coloring with greedy algorithm (degree-descending strategy)"""
    greedy = GreedyColoring(graph)
    coloring, num_colors = greedy.execute()

    # Verify coloring validity
    is_valid = all(coloring[u] != coloring[v] for u, v in graph.edges())

    # Save visualization as PDF if path is provided
    if save_path:
        title = f"Graph Coloring with {num_colors} Colors ({title_suffix})"
        visualize_coloring(graph, coloring, title, save_path)

    return num_colors, coloring, is_valid


def load_graphs_from_data(data_dir='../Data'):
    """Load all graph files from data directory"""
    graphs = []
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    # Iterate through all .pkl files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.pkl'):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'rb') as f:
                    graph = pickle.load(f)
                    # Verify if it's a NetworkX graph object
                    if isinstance(graph, nx.Graph):
                        graphs.append((filename, graph))
                        print(f"Loaded graph: {filename}")
                    else:
                        print(f"Skipped non-graph file: {filename}")
            except Exception as e:
                print(f"Failed to load file {filename}: {str(e)}")

    return graphs


def process_single_graph(filename, graph,visualize=True, save_dir='./coloring_results'):
    """Process a single graph: execute greedy coloring + save visualization as PDF"""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Record execution time
    start_time = time.perf_counter()
    # Execute greedy coloring
    greedy = GreedyColoring(graph)
    coloring, num_colors = greedy.execute()
    # Calculate execution time (convert to milliseconds)
    exec_time = (time.perf_counter() - start_time) * 1000

    # Generate combined visualization and save as PDF
    base_name = os.path.splitext(filename)[0]
    combined_save_path = os.path.join(save_dir, f"{base_name}_combined.pdf")
    visualizer = GraphColoringVisualizer(
        graph=graph,
        coloring=coloring,
        filename=filename,
        num_colors=num_colors,
        exec_time=exec_time
    )
    visualizer.save_combined_visualization(combined_save_path)

    # Save individual coloring result
    coloring_save_path = os.path.join(save_dir, f"{base_name}_coloring.pdf")
    solve_with_greedy(
        graph,
        save_path=coloring_save_path,
        title_suffix="degree descending order"
    )

    # Return results for further analysis
    return {
        "filename": filename,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_colors": num_colors,
        "execution_time_ms": round(exec_time, 2),
        "is_valid": True,
        "save_path": save_dir
    }
