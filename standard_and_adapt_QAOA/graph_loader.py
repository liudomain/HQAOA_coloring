#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图数据加载模块
支持从目录批量加载 .col 和 .pkl 格式图文件

加载策略:
1. format_type='auto' (默认): 优先加载 .col 文件，若无则加载 .pkl 文件
2. format_type='col': 只加载 .col 文件
3. format_type='pkl': 只加载 .pkl 文件

使用方法:
    from graph_loader import load_graphs_from_dir

    # 自动检测并加载（优先 .col，否则 .pkl）
    graphs = load_graphs_from_dir('/path/to/graphs')

    # 强制加载 .col 文件
    graphs = load_graphs_from_dir('/path/to/graphs', format_type='col')

    # 强制加载 .pkl 文件
    graphs = load_graphs_from_dir('/path/to/graphs', format_type='pkl')
"""
import os
import pickle


# ============================================================================
# 格式读取器
# ============================================================================

def read_col_file(file_path):
    """
    读取 .col 格式图文件，返回 NetworkX Graph 对象

    Args:
        file_path: .col 文件路径

    Returns:
        NetworkX Graph 对象
    """
    import networkx as nx

    G = nx.Graph()
    nodes_added = False

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            elif line.startswith('p'):
                parts = line.split()
                if len(parts) >= 4 and parts[1] == 'edge':
                    n = int(parts[2])
                    G.add_nodes_from(range(1, n + 1))
                    nodes_added = True
            elif nodes_added and not line.startswith('p'):
                parts = line.split()
                if len(parts) >= 3 and parts[0] == 'e':
                    u = int(parts[1])
                    v = int(parts[2])
                    G.add_edge(u, v)

    # 如果没有找到头部信息，尝试从边中推断节点
    if not nodes_added and G.number_of_nodes() == 0:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('c') and not line.startswith('p'):
                    parts = line.split()
                    if parts[0] == 'e' and len(parts) >= 3:
                        u = int(parts[1])
                        v = int(parts[2])
                        G.add_edge(u, v)

    return G


def read_pkl_file(file_path):
    """
    读取 .pkl 格式图文件，返回 NetworkX Graph 对象

    Args:
        file_path: .pkl 文件路径

    Returns:
        NetworkX Graph 对象
    """
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)

    # 确保返回的图有 file_name 属性
    if not hasattr(graph, 'file_name'):
        graph.file_name = os.path.basename(file_path)

    return graph


# ============================================================================
# 格式加载器映射表
# ============================================================================

# 格式配置: 扩展名 -> (读取函数, 格式名称)
_FORMAT_READERS = {
    '.col': (read_col_file, '.col'),
    '.pkl': (read_pkl_file, '.pkl')
}


# ============================================================================
# 统一加载函数
# ============================================================================

def _load_graphs_from_dir_with_extension(dir_path, extension, format_name):
    """
    从指定目录加载指定扩展名的图文件（内部函数）

    Args:
        dir_path: 图文件目录路径
        extension: 文件扩展名（如 '.col' 或 '.pkl'）
        format_name: 格式名称（用于输出提示）

    Returns:
        list of NetworkX Graph 对象，每个图带有 file_name 属性
    """
    if extension not in _FORMAT_READERS:
        raise ValueError(f"不支持的文件格式: {extension}")

    reader_func, _ = _FORMAT_READERS[extension]
    graphs = []

    if not os.path.isdir(dir_path):
        print(f"⚠️ 目录不存在: {dir_path}")
        return graphs

    # 获取所有指定扩展名的文件并排序
    all_files = os.listdir(dir_path)
    target_files = sorted([f for f in all_files if f.endswith(extension)])

    if not target_files:
        print(f"⚠️ 目录 {dir_path} 中没有找到 {extension} 文件")
        return graphs

    print(f"从目录加载图数据: {dir_path}")
    print(f"发现 {len(target_files)} 个 {format_name} 文件")

    for idx, filename in enumerate(target_files):
        file_path = os.path.join(dir_path, filename)
        try:
            G = reader_func(file_path)
            G.file_name = filename
            graphs.append(G)
            print(f'  [{idx + 1}/{len(target_files)}] {filename} '
                  f'(节点={G.number_of_nodes()}, 边={G.number_of_edges()})')
        except Exception as e:
            print(f'  ⚠️ 加载文件 {filename} 失败: {e}')

    print(f'✓ 成功加载 {len(graphs)} 张图（格式: {format_name}）')
    return graphs


# ============================================================================
# 公共接口
# ============================================================================

def get_default_data_dir(format_type='auto'):
    """
    获取默认数据目录

    Args:
        format_type: 数据格式类型，影响默认目录选择

    Returns:
        str: 默认数据目录路径
    """
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'Data', 'instances'
    )
    
    # 根据格式类型选择子目录
    if format_type == 'auto':
        return base_dir  # auto 模式先扫描 instances 主目录（优先 .col）
    elif format_type == 'col':
        return base_dir  # .col 文件在 instances 目录下
    else:  # 'pkl'
        return os.path.join(base_dir, 'temp2')  # .pkl 文件在 instances/temp2 目录下


def load_graphs_from_dir(dir_path, format_type='auto'):
    """
    从指定目录加载图文件（支持 .col 和 .pkl 格式）

    加载策略:
    1. format_type='auto' (默认): 优先加载 .col 文件，若无则加载 .pkl 文件
    2. format_type='col': 只加载 .col 文件
    3. format_type='pkl': 只加载 .pkl 文件

    Args:
        dir_path: 图文件目录路径
        format_type: 加载格式类型，可选值: 'auto', 'col', 'pkl' (默认: 'auto')

    Returns:
        list of NetworkX Graph 对象，每个图带有 file_name 属性

    Examples:
        >>> # 自动检测并加载（优先 .col，否则 .pkl）
        >>> graphs = load_graphs_from_dir('/path/to/graphs')

        >>> # 强制加载 .col 文件
        >>> graphs = load_graphs_from_dir('/path/to/graphs', format_type='col')

        >>> # 强制加载 .pkl 文件
        >>> graphs = load_graphs_from_dir('/path/to/graphs', format_type='pkl')
    """
    # 验证 format_type 参数
    valid_formats = ['auto', 'col', 'pkl']
    if format_type not in valid_formats:
        raise ValueError(f"format_type 必须是 {valid_formats} 之一，当前值: {format_type}")

    # 如果 dir_path 为 'default' 或 None，使用默认目录
    if dir_path == 'default' or dir_path is None:
        dir_path = get_default_data_dir(format_type)

    if not os.path.isdir(dir_path):
        print(f"⚠️ 目录不存在: {dir_path}")
        return []

    # 扫描目录中的所有 .col 和 .pkl 文件
    all_files = os.listdir(dir_path)
    col_files = sorted([f for f in all_files if f.endswith('.col')])
    pkl_files = sorted([f for f in all_files if f.endswith('.pkl')])

    # 显示扫描结果
    print(f"📂 扫描目录: {dir_path}")
    if col_files:
        print(f"   发现 {len(col_files)} 个 .col 文件")
    if pkl_files:
        print(f"   发现 {len(pkl_files)} 个 .pkl 文件")

    # 根据格式类型选择加载策略
    if format_type == 'col':
        # 强制加载 .col 文件
        if not col_files:
            print(f"⚠️ 未发现 .col 文件，加载失败")
            print(f"   提示: .col 文件应位于 {get_default_data_dir('col')} 目录")
            return []
        print(f"✓ 强制加载 .col 格式（忽略 {len(pkl_files)} 个 .pkl 文件）")
        return _load_graphs_from_dir_with_extension(dir_path, '.col', '.col')

    elif format_type == 'pkl':
        # 强制加载 .pkl 文件
        if not pkl_files:
            print(f"⚠️ 未发现 .pkl 文件，加载失败")
            print(f"   提示: .pkl 文件应位于 {get_default_data_dir('pkl')} 目录")
            return []
        print(f"✓ 强制加载 .pkl 格式（忽略 {len(col_files)} 个 .col 文件）")
        return _load_graphs_from_dir_with_extension(dir_path, '.pkl', '.pkl')

    else:  # format_type == 'auto'
        # 自动选择：优先加载 .col 文件，若没有则加载 .pkl 文件
        if col_files:
            print(f"✓ 发现 {len(col_files)} 个 .col 文件，优先加载 .col 格式")
            if pkl_files:
                print(f"   将忽略 {len(pkl_files)} 个 .pkl 文件")
            return _load_graphs_from_dir_with_extension(dir_path, '.col', '.col')
        # 否则尝试加载 .pkl 文件
        elif pkl_files:
            print(f"✓ 未发现 .col 文件，将加载 {len(pkl_files)} 个 .pkl 文件")
            return _load_graphs_from_dir_with_extension(dir_path, '.pkl', '.pkl')
        else:
            # 自动模式下，instances 目录没有文件，尝试切换到 temp2 目录加载 pkl
            if 'temp2' not in dir_path:
                temp2_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    'Data', 'instances', 'temp2'
                )
                print(f"📂 当前目录无文件，尝试切换到 temp2 目录...")
                return load_graphs_from_dir(temp2_dir, format_type='auto')
            else:
                print(f"⚠️ 目录 {dir_path} 中没有找到 .col 或 .pkl 文件")
                return []


def load_graphs_from_dir_col(dir_path):
    """
    从指定目录加载所有 .col 格式图文件（强制加载 .col）

    注意：此函数会直接加载 .col 文件，即使目录中同时存在 .pkl 文件
    如需自动选择格式，请使用 load_graphs_from_dir(dir_path, format_type='auto')

    Args:
        dir_path: 图文件目录路径

    Returns:
        list of NetworkX Graph 对象，每个图带有 file_name 属性
    """
    return _load_graphs_from_dir_with_extension(dir_path, '.col', '.col')


def load_graphs_from_pkl_dir(dir_path):
    """
    从指定目录加载所有 .pkl 格式图文件（强制加载 .pkl）

    注意：此函数会直接加载 .pkl 文件，即使目录中同时存在 .col 文件
    如需自动选择格式，请使用 load_graphs_from_dir(dir_path, format_type='auto')

    Args:
        dir_path: 图文件目录路径

    Returns:
        list of NetworkX Graph 对象，每个图带有 file_name 属性
    """
    return _load_graphs_from_dir_with_extension(dir_path, '.pkl', '.pkl')


# ============================================================================
# 默认数据目录（已废弃，使用 get_default_data_dir()）
# ============================================================================

DEFAULT_DATA_DIR = get_default_data_dir('auto')
