#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QAOA算法对比实验工具

支持的算法:
- Adapt-QAOA: 自适应混合算子的量子近似优化算法
- Standard-QAOA: 标准QAOA（使用固定X门混合算子）
- Adapt-QAOA with Noise: 含退极化噪声的自适应QAOA

输出:
- CSV: 算法对比结果

图数据加载策略:
- format_type='auto' (默认): 优先加载 .col 文件，若无则加载 .pkl 文件
- format_type='col': 只加载 .col 文件
- format_type='pkl': 只加载 .pkl 文件
"""
import os
import time
import csv
import argparse
from graph_loader import load_graphs_from_dir

# 延迟导入: 根据用户选择的算法动态导入对应模块
main_adapt = None
main_standard = None
main_adapt_noise = None

# 输出目录配置
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 默认随机种子
SEED = 10


# ============================================================================
# 工具函数
# ============================================================================

def _to_python_native(obj):
    """
    将 numpy 类型转换为 Python 原生类型
    避免写入 CSV 时出现 TypeError
    """
    from numpy import ndarray, int64, float64
    if isinstance(obj, (int64, float64)):
        return obj.item()
    if isinstance(obj, ndarray):
        return obj.tolist()
    return obj


# 动态打补丁: 为 multilevel_adapt_QAOA_k_coloring 添加类型转换函数
import multilevel_adapt_QAOA_k_coloring as mg
if not hasattr(mg, '_to_python_native'):
    mg._to_python_native = _to_python_native


def load_graphs(graph_dir=None, format_type='auto'):
    """
    从指定目录加载图数据

    Args:
        graph_dir: 图数据目录路径，None 则使用默认目录（根据 format_type 自动选择）
        format_type: 加载格式类型，可选值: 'auto', 'col', 'pkl' (默认: 'auto')

    Returns:
        list: NetworkX Graph 对象列表，每个图带有 file_name 属性
    """
    if graph_dir is not None and os.path.isdir(graph_dir):
        return load_graphs_from_dir(graph_dir, format_type=format_type)
    
    # 使用默认目录（graph_loader 会根据 format_type 自动选择）
    return load_graphs_from_dir('default', format_type=format_type)


def save_csv(rows, filename):
    """保存数据到 CSV 文件"""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(rows[0])  # 表头
        writer.writerows(rows[1:])  # 数据行
    print(f'💾 CSV 已保存: {path}')


# ============================================================================
# 实验执行函数
# ============================================================================

def run_single_algorithm(algorithm_func, graph, dataset_name, graph_idx, seed, **kwargs):
    """
    运行单个 QAOA 算法

    Args:
        algorithm_func: 算法主函数 (main_adapt, main_standard, main_adapt_noise)
        graph: NetworkX 图对象
        dataset_name: 数据集名称
        graph_idx: 图索引
        seed: 随机种子
        **kwargs: 传递给算法函数的额外参数

    Returns:
        dict: 包含算法结果的字典
            - unique_colors: 使用颜色数
            - processing_time: 处理时间(秒)
            - success: 是否成功
            - error: 错误信息(失败时)
    """
    result = {
        'unique_colors': -1,
        'processing_time': -1.0,
        'success': False,
        'error': None
    }

    try:
        t0 = time.time()
        algo_results = algorithm_func(
            [graph],
            dataset=dataset_name,
            graph_index=graph_idx,
            seed=seed,
            **kwargs
        )
        result['processing_time'] = round(time.time() - t0, 4)
        result['success'] = True
        result['unique_colors'] = algo_results[0]['unique_colors'] if algo_results else -1
    except Exception as e:
        result['error'] = str(e)
        result['unique_colors'] = -1
        result['processing_time'] = 0.0

    return result


def run_all_algorithms_on_graph(graph, graph_idx, run_adapt, run_standard, run_adapt_noise, noise_prob):
    """
    对单张图运行所有选中的算法

    Args:
        graph: NetworkX 图对象
        graph_idx: 图索引
        run_adapt: 是否运行 Adapt-QAOA
        run_standard: 是否运行 Standard-QAOA
        run_adapt_noise: 是否运行含噪 Adapt-QAOA
        noise_prob: 噪声概率

    Returns:
        dict: 包含所有算法结果的字典
    """
    graph_file = getattr(graph, 'file_name', f'graph_{graph_idx}')
    result = {
        'graph_file': graph_file,
        'graph_index': graph_idx,
        # Adapt-QAOA 结果占位
        'adapt_colors': -1,
        'adapt_time': -1.0,
        'adapt_success': False,
        # Standard-QAOA 结果占位
        'std_colors': -1,
        'std_time': -1.0,
        'std_success': False,
        # 含噪 Adapt-QAOA 结果占位
        'adapt_colors_noise': -1,
        'adapt_time_noise': -1.0,
        'adapt_success_noise': False,
        # 噪声参数
        'noise_prob_used': noise_prob
    }

    # 1. 运行 Adapt-QAOA
    if run_adapt:
        if main_adapt is not None:
            print(f"  ➤ 运行 Adapt-QAOA...")
            adapt_res = run_single_algorithm(
                main_adapt, graph, 'experiment_dataset', graph_idx, SEED
            )
            result['adapt_colors'] = adapt_res['unique_colors']
            result['adapt_time'] = adapt_res['processing_time']
            result['adapt_success'] = adapt_res['success']
            if adapt_res['success']:
                print(f"  ✓ Adapt-QAOA 完成 (颜色数: {adapt_res['unique_colors']}, 耗时: {adapt_res['processing_time']}s)")
            else:
                print(f"  ✗ Adapt-QAOA 失败: {adapt_res['error']}")
        else:
            print(f"  ⚠️ Adapt-QAOA 未导入，请检查 mindquantum 环境")
            result['adapt_success'] = False
            result['error'] = 'Algorithm not imported - check Python environment'

    # 2. 运行 Standard-QAOA
    if run_standard:
        if main_standard is not None:
            print(f"  ➤ 运行 Standard-QAOA...")
            std_res = run_single_algorithm(
                main_standard, graph, 'experiment_dataset', graph_idx, SEED
            )
            result['std_colors'] = std_res['unique_colors']
            result['std_time'] = std_res['processing_time']
            result['std_success'] = std_res['success']
            if std_res['success']:
                print(f"  ✓ Standard-QAOA 完成 (颜色数: {std_res['unique_colors']}, 耗时: {std_res['processing_time']}s)")
            else:
                print(f"  ✗ Standard-QAOA 失败: {std_res['error']}")
        else:
            print(f"  ⚠️ Standard-QAOA 未导入，请检查 mindquantum 环境")
            result['std_success'] = False
            result['error'] = 'Algorithm not imported - check Python environment'

    # 3. 运行含噪 Adapt-QAOA
    if run_adapt_noise:
        if main_adapt_noise is not None:
            print(f"  ➤ 运行含噪 Adapt-QAOA (噪声概率 p={noise_prob})...")
            noise_res = run_single_algorithm(
                main_adapt_noise, graph, 'experiment_dataset', graph_idx, SEED,
                depolarizing_prob=noise_prob
            )
            result['adapt_colors_noise'] = noise_res['unique_colors']
            result['adapt_time_noise'] = noise_res['processing_time']
            result['adapt_success_noise'] = noise_res['success']
            if noise_res['success']:
                print(f"  ✓ 含噪 Adapt-QAOA 完成 (颜色数: {noise_res['unique_colors']}, 耗时: {noise_res['processing_time']}s)")
            else:
                print(f"  ✗ 含噪 Adapt-QAOA 失败: {noise_res['error']}")
        else:
            print(f"  ⚠️ 含噪 Adapt-QAOA 未导入，请检查 mindquantum 环境")
            result['adapt_success_noise'] = False
            result['error'] = 'Algorithm not imported - check Python environment'

    return result


# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='QAOA算法对比实验工具',
        epilog="""
示例用法:
  运行单一算法:
    python run_experiments.py --adapt
    python run_experiments.py --standard
    python run_experiments.py --adapt-noise --noise-prob 0.1

  运行多个算法对比:
    python run_experiments.py --adapt --standard
    python run_experiments.py --adapt --standard --adapt-noise

  使用自定义图目录和格式:
    python run_experiments.py --adapt --graph-dir /path/to/graphs --format-type col
    python run_experiments.py --adapt --graph-dir /path/to/graphs --format-type pkl
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--adapt', action='store_true',
                        help='运行 Adapt-QAOA 算法')
    parser.add_argument('--standard', action='store_true',
                        help='运行 Standard-QAOA 算法')
    parser.add_argument('--adapt-noise', action='store_true',
                        help='运行含噪 Adapt-QAOA 算法')
    parser.add_argument('--noise-prob', type=float, default=0.05,
                        help='含噪实验的噪声概率 (默认: 0.05)')
    parser.add_argument('--seed', type=int, default=10,
                        help='随机种子 (默认: 10)')
    parser.add_argument('--graph-dir', type=str, default=None,
                        help='图数据目录路径 (不指定则使用默认目录)')
    parser.add_argument('--format-type', type=str, default='auto',
                        choices=['auto', 'col', 'pkl'],
                        help='数据加载格式: auto(自动), col(仅.col), pkl(仅.pkl) (默认: auto)')
    return parser.parse_args()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数: 协调整个实验流程"""
    args = parse_args()

    # 更新全局随机种子
    global SEED
    SEED = args.seed

    # 验证: 必须选择至少一个算法
    if not any([args.adapt, args.standard, args.adapt_noise]):
        print("="*70)
        print("⚠️ 错误: 必须选择至少一个算法")
        print("="*70)
        print("\n请使用以下参数:")
        print("  --adapt          运行 Adapt-QAOA")
        print("  --standard       运行 Standard-QAOA")
        print("  --adapt-noise    运行含噪 Adapt-QAOA\n")
        print("示例:")
        print("  python run_experiments.py --adapt --standard")
        print("="*70)
        return

    # 动态导入选中的算法模块
    print("="*70)
    print("正在加载算法模块...")
    print("="*70)
    if args.adapt:
        from Main_Multilevel_qaoa import main_adapt
        global main_adapt
        print('  ✓ 已导入: Adapt-QAOA')
    if args.standard:
        from Main_Multilevel_qaoa import main_standard
        global main_standard
        print('  ✓ 已导入: Standard-QAOA')
    if args.adapt_noise:
        from Main_Multilevel_qaoa import main_adapt_noise
        global main_adapt_noise
        print('  ✓ 已导入: 含噪 Adapt-QAOA')

    # 步骤1: 加载图数据
    print("\n" + "="*70)
    print("步骤 1/3: 加载图数据")
    print("="*70)
    graphs = load_graphs(args.graph_dir, args.format_type)
    if not graphs:
        print("⚠️ 没有加载到任何图数据，程序退出")
        return
    print(f'📦 成功加载 {len(graphs)} 张图 (随机种子: {SEED})')

    # 步骤2: 显示实验配置
    print("\n" + "="*70)
    print("步骤 2/3: 实验配置")
    print("="*70)
    print(f'  随机种子: {SEED}')
    print(f'  噪声概率: {args.noise_prob}')
    print(f'  数据格式: {args.format_type}')
    print(f'  启用的算法:')
    if args.adapt:
        print(f'    ✓ Adapt-QAOA')
    if args.standard:
        print(f'    ✓ Standard-QAOA')
    if args.adapt_noise:
        print(f'    ✓ 含噪 Adapt-QAOA (p={args.noise_prob})')

    # 步骤3: 逐图运行实验
    print("\n" + "="*70)
    print("步骤 3/3: 运行实验")
    print("="*70)
    records = []
    total_start = time.time()

    for idx, graph in enumerate(graphs):
        print(f'\n{"-" * 70}')
        print(f'📊 处理图 {idx}/{len(graphs)-1}: {graph.file_name}')
        print(f'{"-" * 70}')
        records.append(
            run_all_algorithms_on_graph(
                graph, idx, args.adapt, args.standard, args.adapt_noise, args.noise_prob
            )
        )

    total_time = time.time() - total_start

    # 保存综合结果 CSV
    print("\n" + "="*70)
    print("保存结果")
    print("="*70)
    csv_rows = [
        ['graph_file', 'graph_index',
         'adapt_colors', 'adapt_time', 'adapt_success',
         'std_colors', 'std_time', 'std_success',
         'adapt_colors_noise', 'adapt_time_noise', 'adapt_success_noise',
         'noise_prob_used']
    ]
    for r in records:
        csv_rows.append([
            r['graph_file'], r['graph_index'],
            r['adapt_colors'], r['adapt_time'], r['adapt_success'],
            r['std_colors'], r['std_time'], r['std_success'],
            r['adapt_colors_noise'], r['adapt_time_noise'], r['adapt_success_noise'],
            r['noise_prob_used']
        ])
    save_csv(csv_rows, 'all_results.csv')

    # 输出实验总结
    print("\n" + "="*70)
    print("实验完成")
    print("="*70)
    print(f'  输出目录: {os.path.abspath(OUTPUT_DIR)}')
    print(f'  随机种子: {SEED}')
    print(f'  噪声概率: {args.noise_prob}')
    print(f'  处理图数: {len(records)}')
    print(f'  总耗时: {total_time:.2f} 秒')
    print("\n生成的文件:")
    print(f"  📊 all_results.csv   - 实验结果数据")
    print("="*70)


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == '__main__':
    main()



