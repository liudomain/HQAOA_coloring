from collections import deque
from random import randrange,choice
import os
import pickle
import matplotlib
# 使用非交互式后端，图片显示后不阻塞程序继续执行
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import hsv_to_rgb


def tabucol(graph, number_of_colors, tabu_size=7, reps=100,
            max_iterations=10000, debug=False):
    # ===================== 测试语句1：输入参数验证 =====================
    if debug:
        print(f"\n【TabuCol函数启动】输入参数：")
        print(f"- 图节点数：{graph.number_of_nodes()}, 边数：{graph.number_of_edges()}")
        print(f"- 颜色数：{number_of_colors}, 禁忌表大小：{tabu_size}, 采样次数：{reps}")
        print(f"- 最大迭代次数：{max_iterations}")

    # 基础输入检查（保留原逻辑，新增日志）
    if number_of_colors < 1:
        if debug:
            print("【错误】颜色数量必须至少为1（输入值：{}）".format(number_of_colors))
        return None
    if graph.number_of_nodes() == 0:
        if debug:
            print("【提示】输入图为空（节点数=0），直接返回空解")
        return {}

    colors = list(range(number_of_colors))
    iterations = 0
    tabu = deque()
    nodes = list(graph.nodes())
    # 初始化解（新增日志验证节点列表）
    solution = {node: colors[randrange(0, len(colors))] for node in nodes}
    if debug:
        print(f"【初始化】节点列表长度：{len(nodes)}，解字典长度：{len(solution)}")
        print(f"【初始化】颜色列表：{colors}，初始解示例（前5个节点）：{list(solution.items())[:5]}")

    aspiration_level = {}
    last_improve_iter = 0  # 早停计数

    while iterations < max_iterations:
        # ===================== 测试语句2：迭代状态追踪 =====================
        if debug:
            print(f"\n【迭代 {iterations:4d}】当前禁忌表：{list(tabu)}（长度：{len(tabu)}）")
            print(f"【迭代 {iterations:4d}】上一次改进迭代：{last_improve_iter}（间隔：{iterations - last_improve_iter}）")

        move_candidates = set()
        conflict_count = 0

        # 1. 收集冲突节点（易出错点：边遍历中的节点是否在solution中）
        for u, v in graph.edges():
            # 测试：验证u/v是否在solution中（防止KeyError）
            if u not in solution or v not in solution:
                if debug:
                    print(f"【警告】边 ({u},{v}) 包含不在解中的节点（u在解中：{u in solution}，v在解中：{v in solution}）")
                continue  # 跳过无效边
            if solution[u] == solution[v]:
                move_candidates.add(u)
                move_candidates.add(v)
                conflict_count += 1

        if conflict_count == 0:  # 合法解
            if debug:
                print(f"【迭代 {iterations:4d}】找到合法解！冲突数=0，终止搜索")
            break

        # ******** 大图防越界核心（易出错点：move_candidates为空）********
        move_candidates = list(move_candidates)
        if debug:
            print(f"【迭代 {iterations:4d}】收集的冲突节点：{move_candidates}（长度：{len(move_candidates)}）")
        # 测试：若冲突节点为空但有冲突，强制用所有节点（防止后续choice空列表）
        if not move_candidates and conflict_count > 0:
            if debug:
                print(f"【迭代 {iterations:4d}】冲突节点为空但存在冲突，强制使用所有节点作为候选")
            move_candidates = list(graph.nodes())
        # 测试：若候选仍为空，终止（防止choice报错）
        if not move_candidates:
            if debug:
                print(f"【错误】迭代 {iterations:4d}：存在冲突（{conflict_count}个）但候选节点为空，终止搜索")
            break
        # ***********************************

        best_new_solution = None
        best_new_conflicts = float('inf')
        best_move = None
        found_better = False

        # 2. 采样邻居（易出错点：node选择、possible_colors为空）
        if debug:
            print(f"【迭代 {iterations:4d}】开始采样邻居（共{reps}次），候选节点数：{len(move_candidates)}")
        for sample_idx in range(reps):
            # 测试：选择节点前验证候选列表非空（双重保险）
            if not move_candidates:
                if debug:
                    print(f"【错误】迭代 {iterations:4d} 采样 {sample_idx} 次：候选节点为空，跳过本次采样")
                continue
            node = choice(move_candidates)
            # 测试：验证node在solution中（防止KeyError）
            if node not in solution:
                if debug:
                    print(f"【错误】迭代 {iterations:4d} 采样 {sample_idx} 次：选择的节点 {node} 不在解中，跳过")
                continue
            current_color = solution[node]
            possible_colors = [c for c in colors if c != current_color]

            # 测试：possible_colors为空（无可用颜色，跳过）
            if not possible_colors:
                if debug:
                    print(f"【警告】迭代 {iterations:4d} 采样 {sample_idx} 次：节点 {node} 无可用颜色（当前色：{current_color}，总色数：{len(colors)}），跳过")
                continue
            new_color = choice(possible_colors)

            # 测试：生成新解后验证完整性
            new_solution = solution.copy()
            new_solution[node] = new_color
            if len(new_solution) != len(graph.nodes()):
                if debug:
                    print(f"【警告】迭代 {iterations:4d} 采样 {sample_idx} 次：新解不完整（新解节点数：{len(new_solution)}，图节点数：{len(graph.nodes())}）")

            # 增量冲突计算
            new_conflicts = 0
            for u, v in graph.edges():
                # 测试：验证u/v在新解中（防止KeyError）
                if u not in new_solution or v not in new_solution:
                    if debug:
                        print(f"【警告】迭代 {iterations:4d} 采样 {sample_idx} 次：边 ({u},{v}) 包含不在新解中的节点，跳过冲突计算")
                    continue
                if new_solution[u] == new_solution[v]:
                    new_conflicts += 1

            # 测试：打印采样结果（追踪冲突变化）
            if debug and (sample_idx % 20 == 0 or sample_idx == reps-1):  # 每20次或最后一次打印
                print(f"【迭代 {iterations:4d}】采样 {sample_idx:3d} 次：节点 {node} 从色{current_color}→{new_color}，新冲突数：{new_conflicts}（当前最优：{best_new_conflicts}）")

            # 更新最优解
            if new_conflicts < best_new_conflicts:
                best_new_conflicts = new_conflicts
                best_new_solution = new_solution
                best_move = (node, new_color)

            # 检查是否找到更优解（优于当前冲突数）
            if new_conflicts < conflict_count:
                # 愿望水平判断
                if new_conflicts <= aspiration_level.get(conflict_count, conflict_count - 1):
                    aspiration_level[conflict_count] = new_conflicts - 1
                    if (node, new_color) in tabu:
                        if debug:
                            print(f"【迭代 {iterations:4d}】采样 {sample_idx} 次：触发愿望水平，解禁禁忌移动 ({node},{new_color})")
                        found_better = True
                        break
                else:
                    if (node, new_color) in tabu:
                        if debug:
                            print(f"【迭代 {iterations:4d}】采样 {sample_idx} 次：移动 ({node},{new_color}) 在禁忌表中，跳过")
                        continue
                found_better = True
                if debug:
                    print(f"【迭代 {iterations:4d}】采样 {sample_idx} 次：找到更优解（冲突数 {new_conflicts} < {conflict_count}），提前结束采样")
                break

        # 3. 确定最终新解（易出错点：best_new_solution为空）
        if not found_better:
            if best_new_solution is not None:
                node, new_color = best_move
                if debug:
                    print(f"【迭代 {iterations:4d}】未找到更优解，使用当前最优移动：({node},{new_color})，新冲突数：{best_new_conflicts}")
            else:
                best_new_solution = solution.copy()
                if debug:
                    print(f"【迭代 {iterations:4d}】无最优解，保留原解（冲突数：{conflict_count}）")
        elif best_new_solution is None:
            best_new_solution = solution.copy()
            if debug:
                print(f"【迭代 {iterations:4d}】找到更优解但新解为空，保留原解")

        # 4. 更新禁忌表（易出错点：best_move为空）
        if best_move is not None:
            tabu.append(best_move)
            # 控制禁忌表大小
            if len(tabu) > tabu_size:
                removed = tabu.popleft()
                if debug:
                    print(f"【迭代 {iterations:4d}】禁忌表超大小（{len(tabu)+1} > {tabu_size}），移除最早移动：{removed}")
            if debug:
                print(f"【迭代 {iterations:4d}】更新后禁忌表：{list(tabu)}（长度：{len(tabu)}）")
        else:
            if debug:
                print(f"【警告】迭代 {iterations:4d}：best_move为空，未更新禁忌表")

        # 5. 更新解和迭代次数
        solution = best_new_solution
        iterations += 1

        # 6. 早停逻辑（易出错点：迭代次数差值计算）
        if found_better:
            last_improve_iter = iterations
            if debug:
                print(f"【迭代 {iterations:4d}】更新最后改进迭代为：{last_improve_iter}")
        if iterations - last_improve_iter > 500:
            if debug:
                print(f"【迭代 {iterations:4d}】早停触发（500轮无改进），终止搜索")
            break

    # 最终验证（易出错点：solution为空或不完整）
    if debug:
        print(f"\n【搜索结束】总迭代次数：{iterations}，最终解是否存在：{solution is not None}")
        if solution is not None:
            print(f"【搜索结束】最终解节点数：{len(solution)}，图节点数：{len(graph.nodes())}")
    final_conflicts = 0
    if solution is not None:
        final_conflicts = sum(1 for u, v in graph.edges() if u in solution and v in solution and solution[u] == solution[v])
    if debug:
        print(f"【搜索结束】最终冲突数：{final_conflicts}，返回解：{solution is not None}")

    return solution if final_conflicts == 0 else None



def generate_distinct_colors(num_colors):
    """生成一组视觉上明显不同的颜色"""
    if num_colors == 0:
        return []
    hues = np.linspace(0, 1, num_colors, endpoint=False)
    saturation = 0.7
    value = 0.9
    return [hsv_to_rgb((h, saturation, value)) for h in hues]


def visualize_coloring(graph, coloring, title, figsize=(10, 8)):
    """可视化图的着色结果"""
    if not coloring:
        print("没有有效的着色方案可可视化")
        return

    num_colors = len(set(coloring.values()))
    colors = generate_distinct_colors(num_colors)
    node_colors = [colors[coloring[node]] for node in graph.nodes()]

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(graph, seed=42)  # 固定布局

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=300, edgecolors='black')
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=10)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def solve_with_tabu(graph, num_colors, visualize=False, title_suffix="", **kwargs):
    if not isinstance(graph, nx.Graph):
        print("错误：输入必须是NetworkX图对象")
        return None, None, False

    # 开启debug模式（关键：通过debug=True打印测试日志）
    kwargs['debug'] = True  # 强制开启测试日志，定位错误后可改为False
    coloring = tabucol(graph, num_colors, **kwargs)

    is_valid = False
    if coloring is not None:
        try:
            is_valid = all(coloring[u] != coloring[v] for u, v in graph.edges())
        except KeyError as e:
            print(f"着色验证失败：节点 {e} 不在着色方案中")
            is_valid = False

    if visualize and coloring is not None:
        title = f"Tabu Search Coloring with {len(set(coloring.values()))} Colors ({title_suffix})"
        visualize_coloring(graph, coloring, title)

    return coloring, len(set(coloring.values())) if coloring else None, is_valid

def load_graphs_from_data(data_dir='../Data'):
    graphs = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录 {data_dir} 不存在")

    for filename in os.listdir(data_dir):
        if filename.endswith('.pkl'):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'rb') as f:
                    graph = pickle.load(f)

                    if not isinstance(graph, nx.Graph):
                        print(f"跳过非图文件: {filename}")
                        continue

                    if graph.number_of_nodes() == 0:
                        print(f"跳过空图: {filename}")
                        continue

                    if graph.number_of_edges() == 0:
                        print(f"注意：{filename} 是孤立图（没有边）")

                    graphs.append((filename, graph))
                    print(f"已加载图: {filename} (节点: {graph.number_of_nodes()}, 边: {graph.number_of_edges()})")

            except Exception as e:
                print(f"加载文件 {filename} 失败: {str(e)}")

    return graphs

def estimate_chromatic_number(graph):
    """估算图的色数（用于设置初始颜色数量）"""
    if graph.number_of_nodes() == 0:
        return 0
    # 色数至少为最大团大小，这里用最大度数加1作为保守估计
    max_degree = max(dict(graph.degree()).values(), default=0)
    return max_degree + 1


if __name__ == "__main__":
    try:
        # 测试时可先加载1个小图，避免日志过多
        graph_list = load_graphs_from_data('../Data')
        if not graph_list:
            print("未找到任何图数据，请先运行数据集生成程序")
        else:
            # 仅测试第一个图（定位错误时减少干扰）
            filename, graph = graph_list[0]
            print(f"\n===== 测试单个图: {filename} =====")
            print(f"节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")

            estimated_k = estimate_chromatic_number(graph)
            print(f"估算色数: {estimated_k}")

            if graph.number_of_edges() == 0:
                print("孤立图：所有节点可使用同一种颜色")
                solve_with_tabu(
                    graph,
                    1,
                    visualize=True,
                    title_suffix="isolated graph"
                )
            else:
                coloring, num_colors, is_valid = solve_with_tabu(
                    graph,
                    estimated_k,
                    visualize=True,
                    title_suffix="tabu search test",
                    max_iterations=1000,  # 测试时减少迭代次数，快速定位
                    tabu_size=5,
                    reps=50,
                    debug=True  # 强制开启测试日志
                )

                if coloring:
                    print(f"禁忌搜索: 找到有效着色，使用 {num_colors} 种颜色，着色{'' if is_valid else '不'}有效")
                else:
                    print(f"禁忌搜索: 在 {estimated_k} 种颜色下未找到有效着色方案")

    except Exception as e:
        # 捕获全局异常，打印错误位置
        import traceback
        print(f"\n【全局错误】处理过程出错: {str(e)}")
        print("【错误堆栈】")
        traceback.print_exc()  # 打印详细堆栈信息，定位出错行
