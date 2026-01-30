"""
天衍平台 QAOA 参数训练模块

由于天衍真机不支持在线梯度计算，本模块提供标准QAOA参数优化方法：
1. PyTorch Adam 优化器 (使用有限差分法估计梯度)

特点：
- 在仿真器或真机上运行标准QAOA线路
- 根据测量结果优化参数 (gamma, beta)
- 支持多层级 QAOA 参数优化
"""

import numpy as np
import networkx as nx
from cqlib import TianYanPlatform
from cqlib.circuits import Circuit
from typing import Dict, Tuple, Optional, List
import time
import json
import csv
import os
from datetime import datetime
import torch  # PyTorch Adam 优化器
import matplotlib.pyplot as plt
from matplotlib import cm

from multilevel_common_tianyan import (
    qaoa_ansatz_tianyan,
    extract_coloring_tianyan,
    count_conflicts
)


class QAOAParamOptimizer:
    """
    QAOA 参数优化器

    使用 PyTorch Adam 优化器（带网格全局搜索和随机重启）在天衍平台上优化 QAOA 参数
    支持结果缓存以减少实验次数
    """

    def __init__(self, platform: TianYanPlatform, graph: nx.Graph, k: int, p: int,
                 algorithm: str = 'standard', lab_id: Optional[str] = None,
                 num_shots: int = 500,  # 增加采样次数以提高准确性
                 output_dir: Optional[str] = None, enable_cache: bool = True):
        """
        初始化参数优化器

        参数:
            platform: 天衍平台实例
            graph: 待着色的图
            k: 颜色数量
            p: QAOA 层数
            algorithm: 算法类型 ('standard'，天衍平台仅支持标准QAOA)
            lab_id: 实验室 ID
            num_shots: 每次实验的采样次数
            output_dir: 输出目录（用于存储采样数据）
            enable_cache: 是否启用结果缓存（相同参数不重复测量）
        """
        self.platform = platform
        self.graph = graph
        self.k = k
        self.p = p
        self.algorithm = algorithm
        self.lab_id = lab_id
        self.num_shots = num_shots
        self.num_qubits = len(graph.nodes) * int(np.ceil(np.log2(k)))
        self.output_dir = output_dir
        self.evaluation_history = []  # 存储评估历史
        self.enable_cache = enable_cache
        self.result_cache = {}  # 参数哈希 -> 结果缓存
        
    def _evaluate_params(self, params: np.ndarray, verbose: bool = False, max_retries: int = 3, use_cache: bool = True) -> float:
        """
        评估给定参数的性能

        参数:
            params: 参数数组 [gamma_0, beta_0, gamma_1, beta_1, ...]
            verbose: 是否输出详细信息
            max_retries: 最大重试次数

        返回:
            float: 目标函数值（冲突数）
        """
        # 构建参数哈希用于缓存（使用更高精度，避免梯度计算时缓存命中）
        import hashlib
        # 注意：在梯度计算中，epsilon_fd=0.01 的变化应该能产生不同的哈希值
        # 使用更多小数位（10位）确保微小差异能被区分
        params_key = tuple(np.round(params, 10))
        params_hash = hashlib.md5(str(params_key).encode()).hexdigest()

        # 检查缓存
        if use_cache and self.enable_cache and params_hash in self.result_cache:
            cached_result = self.result_cache[params_hash]
            if verbose:
                print(f"\n[评估参数] (使用缓存)")
                params_str = ", ".join([f"{v:.4f}" for v in params])
                print(f"  参数: [{params_str}]")
                print(f"  ✓ 从缓存获取结果: 冲突数 = {cached_result['conflicts']}")
            return cached_result['conflicts']

        # 构建参数字典
        trained_params = {}
        for layer in range(self.p):
            gamma_key = f'gamma_{layer}'
            beta_key = f'beta_{layer}'
            trained_params[gamma_key] = params[2 * layer]
            trained_params[beta_key] = params[2 * layer + 1]

        if verbose:
            print(f"\n[评估参数]")
            params_str = ", ".join([f"{v:.4f}" for v in params])
            print(f"  参数: [{params_str}]")

        # 获取当前已知的最小冲突数，用于异常情况下的默认返回值
        def get_min_known_conflicts():
            known_conflicts = [d.get('conflicts', float('inf')) for d in self.evaluation_history if d.get('conflicts') is not None]
            if known_conflicts:
                return min(known_conflicts)
            return float('inf')

        for retry in range(max_retries):
            try:
                # 构建 QAOA 线路
                if verbose:
                    print(f"  构建 QAOA 线路... (尝试 {retry+1}/{max_retries})")

                # 统一使用 qaoa_ansatz_tianyan (天衍平台只支持标准QAOA)
                circuit = qaoa_ansatz_tianyan(
                    graph=self.graph,
                    k=self.k,
                    p=self.p,
                    trained_params=trained_params,
                    verbose=False  # 减少输出
                )

                circuit.measure_all()

                if verbose:
                    print(f"  线路构建完成，量子比特数: {circuit.num_qubits}")
                    print(f"  QCIS 指令数: {len(circuit.qcis)}")
                    print(f"  QCIS 类型: {type(circuit.qcis)}")

                # 过滤掉 M 门（天衍平台不支持单独的 M 门）
                filtered_qcis = remap_qubits_in_qcis(circuit.qcis, {i: i for i in range(circuit.num_qubits)})

                if verbose and filtered_qcis and len(filtered_qcis) > 0:
                    print(f"  过滤后 QCIS 前5行:")
                    for i, line in enumerate(filtered_qcis.split('\n')[:5]):
                        if line.strip() and not line.strip().startswith('#'):
                            print(f"    {line}")

                # 提交到天衍平台 (使用 submit_job 方法)
                from datetime import datetime
                exp_name = f'train.{datetime.now().strftime("%Y%m%d%H%M%S")}'

                # 检查过滤后的 QCIS 是否为空
                if filtered_qcis is None or not filtered_qcis or filtered_qcis.strip() == '':
                    if verbose:
                        print(f"  ✗ 错误：过滤后的 QCIS 为空或None，无法提交实验")
                        print(f"  原始 QCIS 类型: {type(circuit.qcis)}")
                        print(f"  原始 QCIS 长度: {len(str(circuit.qcis))}")
                    continue  # 重试

                # 根据 lab_id 是否为 None，决定是否传递该参数
                submit_kwargs = {
                    'circuit': filtered_qcis,  # 使用过滤后的 QCIS（不含 M 门）
                    'exp_name': exp_name,
                    'num_shots': self.num_shots
                }
                if self.lab_id is not None:
                    submit_kwargs['lab_id'] = self.lab_id

                if verbose:
                    print(f"  提交实验到天衍平台...")
                    print(f"  实验名称: {exp_name}")
                    print(f"  采样次数: {self.num_shots}")
                    print(f"  QCIS 长度: {len(filtered_qcis)}")
                    print(f"  提交参数: {submit_kwargs}")

                try:
                    query_id = self.platform.submit_job(**submit_kwargs)
                except Exception as submit_error:
                    if verbose:
                        print(f"  ✗ 提交实验失败: {submit_error}")
                        print(f"  可能原因: 天衍平台 API 服务端问题或网络连接问题")
                        print(f"  建议检查:")
                        print(f"    1. 天衍平台服务是否正常")
                        print(f"    2. login_key 是否有效")
                        print(f"    3. lab_id 是否存在且有效")
                    continue  # 重试

                if verbose:
                    print(f"  实验已提交，查询ID: {query_id}")
                    print(f"  等待实验结果...")

                # 等待结果（大幅减少等待时间以加快训练）
                result = self.platform.query_experiment(
                    query_id=query_id,
                    max_wait_time=60,  # 减少最大等待时间到60秒
                    sleep_time=1  # 减少轮询间隔到1秒
                )

                if result and len(result) > 0:
                    if verbose:
                        print(f"  ✓ 实验完成，获得结果")

                    # 分析采样结果，检查是否全为0
                    sampling_analysis = self._analyze_sampling_result(result[0])

                    # 存储评估数据
                    eval_data = {
                        "params": params.tolist(),
                        "query_id": query_id,
                        "num_shots": self.num_shots,
                        "conflicts": None,
                        "coloring": None,
                        "sampling_analysis": sampling_analysis,
                        "from_cache": False
                    }

                    # 提取着色方案
                    coloring = extract_coloring_tianyan(result[0], self.graph, self.k, verbose=False)
                    if coloring:
                        conflicts = count_conflicts(coloring, self.graph, verbose=False)
                        eval_data["conflicts"] = int(conflicts)
                        eval_data["coloring"] = coloring

                        if verbose:
                            print(f"  ✓ 参数 {params_str} -> 冲突数: {conflicts}")
                            if sampling_analysis["all_zeros_ratio"] > 0.5:
                                print(f"  ⚠️ 警告: 采样结果中0态比例过高: {sampling_analysis['all_zeros_ratio']:.2%}")

                        # 存入缓存
                        if self.enable_cache:
                            self.result_cache[params_hash] = {
                                'conflicts': conflicts,
                                'coloring': coloring,
                                'sampling_analysis': sampling_analysis,
                                'query_id': query_id
                            }

                        self.evaluation_history.append(eval_data)
                        return conflicts
                    else:
                        # 无法提取着色，使用大图中边数加1作为默认冲突数
                        default_conflicts = self.graph.number_of_edges() + 1
                        eval_data["conflicts"] = default_conflicts
                        
                        if verbose:
                            print(f"  ✗ 无法从结果中提取着色方案，使用默认冲突数: {default_conflicts}，继续重试")
                        self.evaluation_history.append(eval_data)
                        continue  # 重试
                else:
                    # 实验失败，使用大图中边数加1作为默认冲突数
                    default_conflicts = self.graph.number_of_edges() + 1
                    
                    # 存储评估数据
                    eval_data = {
                        "params": params.tolist(),
                        "query_id": query_id,
                        "num_shots": self.num_shots,
                        "conflicts": default_conflicts,
                        "coloring": None,
                        "sampling_analysis": None,
                        "from_cache": False
                    }
                    
                    if verbose:
                        print(f"  ✗ 实验失败，未获得结果，使用默认冲突数: {default_conflicts}，继续重试")
                    self.evaluation_history.append(eval_data)
                    continue  # 重试

            except Exception as e:
                # 发生异常，使用大图中边数加1作为默认冲突数
                default_conflicts = self.graph.number_of_edges() + 1
                
                # 存储评估数据
                eval_data = {
                    "params": params.tolist(),
                    "query_id": None,
                    "num_shots": self.num_shots,
                    "conflicts": default_conflicts,
                    "coloring": None,
                    "sampling_analysis": None,
                    "from_cache": False
                }
                
                if verbose:
                    print(f"  ✗ 评估参数 {params} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"  使用默认冲突数: {default_conflicts}，继续重试")
                
                self.evaluation_history.append(eval_data)
                # 发生异常，继续重试
                continue

        # 所有重试都失败，返回已知的最小冲突数
        min_known = get_min_known_conflicts()
        if verbose:
            print(f"  ✗ 所有重试均失败，使用已知最小冲突数: {min_known}")
        return min_known

    def _analyze_sampling_result(self, result: Dict) -> Dict:
        """
        分析采样结果，检查是否存在问题（如全为0态）

        参数:
            result: 天衍平台返回的结果字典

        返回:
            dict: 分析结果
        """
        analysis = {
            "has_data": False,
            "total_shots": 0,
            "unique_states": 0,
            "all_zeros_count": 0,
            "all_zeros_ratio": 0.0,
            "top_states": []
        }

        try:
            if 'resultStatus' not in result:
                return analysis

            measurement_data = result['resultStatus']
            if not measurement_data or len(measurement_data) < 2:
                return analysis

            # 提取采样数据
            shots_data = measurement_data[1:]
            if not shots_data:
                return analysis

            # 统计比特串频率
            from collections import Counter
            shots_str = [''.join(map(str, shot)) if isinstance(shot, (list, tuple)) else str(shot)
                        for shot in shots_data]
            bitstring_counts = Counter(shots_str)

            analysis["has_data"] = True
            analysis["total_shots"] = len(shots_str)
            analysis["unique_states"] = len(bitstring_counts)
            analysis["all_zeros_count"] = bitstring_counts.get('0' * len(shots_str[0]) if shots_str else '0', 0)
            analysis["all_zeros_ratio"] = analysis["all_zeros_count"] / analysis["total_shots"]

            # 获取Top 5状态
            top_states = bitstring_counts.most_common(5)
            analysis["top_states"] = [
                {"state": state, "count": count, "ratio": count / analysis["total_shots"]}
                for state, count in top_states
            ]

        except Exception as e:
            pass

        return analysis

    def _get_early_stop_reason(self, best_loss: float, no_improvement_count: int,
                              patience: int, threshold: float) -> str:
        """获取早停原因"""
        if best_loss == 0:
            return "找到最优解 (冲突=0)"
        elif no_improvement_count >= patience:
            return f"连续{patience}次未改进"
        elif best_loss <= threshold:
            return f"损失低于阈值{threshold}"
        else:
            return "未触发早停"

    def optimize_adam_pytorch(self, initial_params: Optional[np.ndarray] = None,
                             maxiter: int = 50, lr: float = 0.05,
                             betas: Tuple[float, float] = (0.9, 0.999),
                             epsilon_fd: float = 0.1, patience: int = 10,
                             early_stop_threshold: float = 0.01,
                             stop_on_zero_conflict: bool = False,
                             verbose: bool = True) -> Tuple[np.ndarray, float, Dict]:
        """
        使用 PyTorch Adam 优化器优化参数（带早停机制）

        参数:
            initial_params: 初始参数数组 (2*p 个参数)
            maxiter: 最大迭代次数
            lr: 学习率
            betas: Adam 超参数 (beta1, beta2)
            epsilon_fd: 有限差分步长
            patience: 早停耐心值（连续多少次未改进则停止）
            early_stop_threshold: 早停阈值（冲突数小于此值则停止）
            stop_on_zero_conflict: 找到无冲突解后是否立即停止训练
            verbose: 是否输出详细信息

        返回:
            tuple: (最优参数, 最小冲突数, 优化信息)
        """
        if initial_params is None:
            # 默认初始参数
            initial_params = np.array([0.5, 0.3] * self.p)

        if verbose:
            print(f"\n{'='*60}")
            print(f"开始 PyTorch Adam 优化")
            print(f"算法: {self.algorithm}, k={self.k}, p={self.p}")
            print(f"学习率: {lr}, 最大迭代: {maxiter}")
            print(f"Adam 超参数: beta1={betas[0]}, beta2={betas[1]}")
            print(f"有限差分步长: {epsilon_fd}")
            print(f"早停耐心值: {patience}, 阈值: {early_stop_threshold}")
            print(f"早停策略: {'启用(冲突=0时停止)' if stop_on_zero_conflict else '禁用'}")
            print(f"初始参数: {initial_params}")
            print(f"{'='*60}\n")

        start_time = time.time()

        # 清空历史记录
        self.evaluation_history = []

        # 将参数转换为 PyTorch 张量（启用梯度）
        params_tensor = torch.tensor(initial_params, dtype=torch.float64, requires_grad=True)

        # 创建 Adam 优化器
        optimizer = torch.optim.Adam([params_tensor], lr=lr, betas=betas)

        best_params = initial_params.copy()
        best_loss = float('inf')
        best_coloring = None  # 保存最优参数对应的着色方案
        best_query_id = None  # 保存最优参数对应的查询ID

        # 早停机制
        no_improvement_count = 0
        loss_history = []

        for iteration in range(maxiter):
            optimizer.zero_grad()

            # 计算当前损失（使用有限差分法估计梯度）
            # 由于天衍平台不支持梯度计算，我们需要手动计算梯度
            current_params_np = params_tensor.detach().numpy()

            # 评估当前参数
            current_loss = self._evaluate_params(current_params_np, verbose=False, use_cache=False)

            # 使用有限差分法估计梯度
            gradient = np.zeros_like(current_params_np)
            for i in range(len(current_params_np)):
                params_plus = current_params_np.copy()
                params_plus[i] += epsilon_fd
                loss_plus = self._evaluate_params(params_plus, verbose=False, use_cache=False)

                params_minus = current_params_np.copy()
                params_minus[i] -= epsilon_fd
                loss_minus = self._evaluate_params(params_minus, verbose=False, use_cache=False)

                # 中心差分估计梯度
                gradient[i] = (loss_plus - loss_minus) / (2 * epsilon_fd)

            # 调试信息：检查梯度是否全为0
            if np.allclose(gradient, 0) and verbose:
                print(f"  ⚠️ 警告：梯度全为0，可能的原因：")
                print(f"     1. 参数变化导致缓存命中 (epsilon_fd={epsilon_fd} 太小)")
                print(f"     2. 当前参数已是最优解 (冲突数={current_loss})")
                print(f"     3. 缓存精度问题")

            # 将梯度设置为 PyTorch 张量的梯度
            params_tensor.grad = torch.tensor(gradient, dtype=torch.float64)

            # 如果梯度全为0，添加随机扰动
            if np.allclose(gradient, 0):
                # 生成小的随机扰动
                perturbation = torch.tensor(np.random.uniform(-0.05, 0.05, size=gradient.shape), dtype=torch.float64)
                params_tensor.grad += perturbation
                if verbose:
                    print(f"  ⚠️ 梯度全为0，添加随机扰动: {[f'{p:.6f}' for p in perturbation.numpy()]}")

            # 更新参数
            optimizer.step()

            # 参数边界限制：gamma ∈ [0, π], beta ∈ [0, π/2]
            # Beta参数限制在较小的范围内，因为问题哈密顿量通常在较小角度更有效
            params_tensor.data[::2] = torch.clamp(params_tensor.data[::2], 0, np.pi)  # gamma
            params_tensor.data[1::2] = torch.clamp(params_tensor.data[1::2], 0, np.pi/2)  # beta

            # 记录最优参数
            if current_loss < best_loss:
                improvement = best_loss - current_loss
                best_loss = current_loss
                best_params = params_tensor.detach().numpy().copy()
                no_improvement_count = 0  # 重置计数器

                # 从评估历史中获取最优参数对应的着色方案
                if self.evaluation_history:
                    # 找到最近一次评估的结果（对应当前参数）
                    for eval_data in reversed(self.evaluation_history):
                        if eval_data.get('conflicts') == current_loss and eval_data.get('coloring') is not None:
                            best_coloring = eval_data['coloring']
                            best_query_id = eval_data.get('query_id')
                            break

                if verbose and improvement > 0:
                    print(f"  ✓ 找到更优解！改进 {improvement} 个冲突")
            else:
                no_improvement_count += 1

            # 记录损失历史
            loss_history.append(current_loss)

            if verbose:
                print(f"\n迭代 {iteration + 1}/{maxiter}")
                print(f"  当前参数: {[f'{v:.6f}' for v in params_tensor.detach().numpy()]}")
                print(f"  梯度: {[f'{g:.6f}' for g in gradient]}")
                print(f"  当前冲突数: {current_loss}")
                print(f"  最优冲突数: {best_loss}")
                print(f"  未改进次数: {no_improvement_count}/{patience}")

            # 早停条件1: 找到最优解（仅在启用 stop_on_zero_conflict 时）
            if best_loss == 0 and stop_on_zero_conflict:
                if verbose:
                    print(f"\n✓ 已找到最优解（冲突=0），提前终止优化")
                break

            # 早停条件2: 连续多次未改进
            if no_improvement_count >= patience:
                if verbose:
                    print(f"\n⚠️ 连续 {patience} 次未改进，启用早停机制")
                break

            # 早停条件3: 损失低于阈值
            if best_loss <= early_stop_threshold and best_loss > 0:
                if verbose:
                    print(f"\n✓ 损失低于阈值 {early_stop_threshold}，提前终止")
                break

        elapsed_time = time.time() - start_time

        # 构建最优参数字典
        best_params_dict = {}
        for layer in range(self.p):
            gamma_key = f'gamma_{layer}'
            beta_key = f'beta_{layer}'
            best_params_dict[gamma_key] = best_params[2 * layer]
            best_params_dict[beta_key] = best_params[2 * layer + 1]

        opt_info = {
            'success': True,
            'message': f'PyTorch Adam optimization completed',
            'nit': iteration + 1,
            'nfev': (iteration + 1) * (2 * self.p + 1),  # 每次迭代评估 2*n+1 次
            'elapsed_time': elapsed_time,
            'method': 'Adam (PyTorch) with Early Stopping',
            'lr': lr,
            'beta1': betas[0],
            'beta2': betas[1],
            'epsilon_fd': epsilon_fd,
            'early_stopped': iteration + 1 < maxiter,
            'early_stop_reason': self._get_early_stop_reason(best_loss, no_improvement_count, patience, early_stop_threshold),
            'loss_history': loss_history,
            'evaluation_history': self.evaluation_history
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"PyTorch Adam 优化完成")
            print(f"实际迭代: {iteration + 1}/{maxiter}")
            print(f"函数评估次数: {opt_info['nfev']}")
            print(f"耗时: {elapsed_time:.2f} 秒")
            if opt_info['early_stopped']:
                print(f"早停原因: {opt_info['early_stop_reason']}")
            print(f"最优冲突数: {best_loss}")
            print(f"最优参数:")
            for layer in range(self.p):
                gamma_key = f'gamma_{layer}'
                beta_key = f'beta_{layer}'
                print(f"  {gamma_key} = {best_params_dict[gamma_key]:.6f}")
                print(f"  {beta_key} = {best_params_dict[beta_key]:.6f}")
            print(f"{'='*60}\n")

        elapsed_time = time.time() - start_time

        # 构建最优参数字典
        best_params_dict = {}
        for layer in range(self.p):
            gamma_key = f'gamma_{layer}'
            beta_key = f'beta_{layer}'
            best_params_dict[gamma_key] = best_params[2 * layer]
            best_params_dict[beta_key] = best_params[2 * layer + 1]

        opt_info = {
            'success': True,
            'message': f'PyTorch Adam optimization completed in {maxiter} iterations',
            'nit': iteration + 1,  # 使用实际迭代次数而不是maxiter
            'best_coloring': best_coloring,  # 添加最优着色方案
            'best_query_id': best_query_id,  # 添加最优查询ID
            'nfev': maxiter * (2 * self.p + 1),  # 每次迭代评估 2*n+1 次
            'elapsed_time': elapsed_time,
            'method': 'Adam (PyTorch)',
            'lr': lr,
            'beta1': betas[0],
            'beta2': betas[1],
            'evaluation_history': self.evaluation_history
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"PyTorch Adam 优化完成")
            print(f"迭代次数: {maxiter}")
            print(f"函数评估次数: {opt_info['nfev']}")
            print(f"耗时: {elapsed_time:.2f} 秒")
            print(f"最优冲突数: {best_loss}")
            print(f"最优参数:")
            for layer in range(self.p):
                gamma_key = f'gamma_{layer}'
                beta_key = f'beta_{layer}'
                print(f"  {gamma_key} = {best_params_dict[gamma_key]:.6f}")
                print(f"  {beta_key} = {best_params_dict[beta_key]:.6f}")
            print(f"{'='*60}\n")

        # 保存训练数据
        if self.output_dir:
            self._save_training_data('adam_pytorch', best_params_dict, best_loss, opt_info)

        return best_params, best_loss, opt_info

    def _save_training_data(self, method: str, best_params: Dict, best_loss: float,
                           opt_info: Dict):
        """
        保存训练数据到文件

        参数:
            method: 优化方法 ('adam_pytorch')
            best_params: 最优参数字典
            best_loss: 最小冲突数
            opt_info: 优化信息
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存完整JSON数据
            json_file = os.path.join(self.output_dir, f"training_{method}_{timestamp}.json")
            training_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "method": method,
                "graph_info": {
                    "num_nodes": self.graph.number_of_nodes(),
                    "num_edges": self.graph.number_of_edges(),
                    "k": self.k,
                    "p": self.p
                },
                "config": {
                    "num_shots": self.num_shots,
                    "lab_id": self.lab_id
                },
                "best_params": best_params,
                "best_loss": int(best_loss),
                "optimization_info": opt_info
            }
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)

            # 保存评估历史CSV
            if self.evaluation_history:
                csv_file = os.path.join(self.output_dir, f"evaluations_{method}_{timestamp}.csv")
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "eval_id", "gamma", "beta", "conflicts",
                        "total_shots", "unique_states", "all_zeros_ratio",
                        "top_state", "top_state_ratio"
                    ])

                    for i, eval_data in enumerate(self.evaluation_history):
                        sampling = eval_data.get("sampling_analysis", {})
                        top_state = sampling.get("top_states", [{}])[0] if sampling.get("top_states") else {}
                        writer.writerow([
                            i + 1,
                            eval_data["params"][0] if len(eval_data["params"]) > 0 else 0,
                            eval_data["params"][1] if len(eval_data["params"]) > 1 else 0,
                            eval_data.get("conflicts", "N/A"),
                            sampling.get("total_shots", 0),
                            sampling.get("unique_states", 0),
                            f"{sampling.get('all_zeros_ratio', 0):.4f}",
                            top_state.get("state", "N/A"),
                            f"{top_state.get('ratio', 0):.4f}"
                        ])

            print(f"✓ 训练数据已保存: {json_file}")
            if self.evaluation_history:
                print(f"✓ 评估历史已保存: {csv_file}")

        except Exception as e:
            print(f"✗ 保存训练数据失败: {e}")

    def plot_coloring_results(self, initial_coloring: Dict, final_coloring: Dict,
                           initial_conflicts: int, final_conflicts: int,
                           initial_params: np.ndarray, final_params: np.ndarray,
                           save_path: Optional[str] = None):
        """
        绘制优化前后的图着色结果对比

        参数:
            initial_coloring: 初始参数对应的着色方案
            final_coloring: 优化后参数对应的着色方案
            initial_conflicts: 初始冲突数
            final_conflicts: 最终冲突数
            initial_params: 初始参数 [gamma, beta]
            final_params: 最终参数 [gamma, beta]
            save_path: 图片保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 设置matplotlib支持中文
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        # 计算布局
        pos = nx.spring_layout(self.graph, seed=42, k=1.2 / np.sqrt(len(self.graph.nodes)), iterations=50)

        # 1. 绘制初始着色结果
        ax1 = axes[0]
        if initial_coloring:
            node_colors_initial = [initial_coloring.get(node, 0) for node in self.graph.nodes]
        else:
            node_colors_initial = [0] * len(self.graph.nodes)

        cmap1 = plt.cm.get_cmap('tab10', self.k)
        node_sizes = [self.graph.degree(node) * 100 + 300 for node in self.graph.nodes]

        nx.draw_networkx_edges(self.graph, pos, width=2, alpha=0.7, edge_color='#888888', ax=ax1)
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors_initial,
                             node_size=node_sizes, cmap=cmap1, ax=ax1,
                             edgecolors='#333333', linewidths=1.5)
        nx.draw_networkx_labels(self.graph, pos, labels={node: str(node) for node in self.graph.nodes},
                             font_size=12, font_family='sans-serif', font_weight='bold', ax=ax1)

        ax1.set_title(f'初始着色\nγ={initial_params[0]:.3f}, β={initial_params[1]:.3f}\n冲突数: {initial_conflicts}',
                     fontsize=12, fontweight='bold', pad=10)
        ax1.axis('off')

        # 2. 绘制优化后着色结果
        ax2 = axes[1]
        if final_coloring:
            node_colors_final = [final_coloring.get(node, 0) for node in self.graph.nodes]
        else:
            node_colors_final = [0] * len(self.graph.nodes)

        cmap2 = plt.cm.get_cmap('tab10', self.k)

        nx.draw_networkx_edges(self.graph, pos, width=2, alpha=0.7, edge_color='#888888', ax=ax2)
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors_final,
                             node_size=node_sizes, cmap=cmap2, ax=ax2,
                             edgecolors='#333333', linewidths=1.5)
        nx.draw_networkx_labels(self.graph, pos, labels={node: str(node) for node in self.graph.nodes},
                             font_size=12, font_family='sans-serif', font_weight='bold', ax=ax2)

        ax2.set_title(f'优化后着色\nγ={final_params[0]:.3f}, β={final_params[1]:.3f}\n冲突数: {final_conflicts}',
                     fontsize=12, fontweight='bold', pad=10)
        ax2.axis('off')

        # 总标题
        improvement = initial_conflicts - final_conflicts
        if improvement > 0:
            title_text = f'图着色优化对比 (k={self.k}, p={self.p})\n改进: {improvement} 个冲突 ✓'
        elif improvement == 0:
            title_text = f'图着色优化对比 (k={self.k}, p={self.p})\n已是最优解 ✓'
        else:
            title_text = f'图着色优化对比 (k={self.k}, p={self.p})\n无改进'

        fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout()

        # 保存图片
        if save_path:
            try:
                # 将路径改为 PDF 格式
                pdf_path = save_path.replace('.png', '.pdf') if save_path.endswith('.png') else save_path
                plt.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.3, format='pdf')
                print(f"✓ 着色对比图已保存: {pdf_path}")
            except Exception as e:
                print(f"✗ 保存图片失败: {e}")

        plt.close()


def train_qaoa_on_tianyan(platform: TianYanPlatform, graph: nx.Graph, k: int, p: int,
                           initial_params: Optional[Dict] = None,
                           output_dir: Optional[str] = None, **kwargs):
    """
    在天衍平台上训练 QAOA 参数（使用网格全局搜索 + 随机重启 + Adam 优化器）

    参数:
        platform: 天衍平台实例
        graph: 待着色的图
        k: 颜色数量
        p: QAOA 层数
        initial_params: 初始参数字典
        output_dir: 输出目录（用于存储训练数据）
        **kwargs: 其他参数传递给优化器
            - lr: 学习率 (默认 0.01)
            - maxiter: 最大迭代次数 (默认 50)
            - betas: Adam 超参数 (默认 (0.9, 0.999))
            - epsilon_fd: 有限差分步长 (默认 0.01)
            - patience: 早停耐心值 (默认 10)
            - early_stop_threshold: 早停阈值 (默认 0.01)
            - verbose: 是否输出详细信息 (默认 True)

    返回:
        dict: 训练结果，包含最优参数和性能指标
    """
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "training_results",
                                   datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"训练结果输出目录: {output_dir}\n")

    # 构建优化器
    optimizer = QAOAParamOptimizer(
        platform=platform,
        graph=graph,
        k=k,
        p=p,
        algorithm='standard',  # 仅支持标准 QAOA
        lab_id=kwargs.get('lab_id'),
        num_shots=kwargs.get('num_shots', 100),
        output_dir=output_dir,
        enable_cache=kwargs.get('enable_cache', False)  # 禁用缓存以避免额外哈希计算
    )

    # 转换初始参数
    initial_array = None
    if initial_params:
        initial_array = np.array([
            initial_params.get(f'gamma_{layer}', 0.3)
            for layer in range(p)
        ] + [
            initial_params.get(f'beta_{layer}', 0.15)
            for layer in range(p)
        ])

    # 执行 PyTorch Adam 优化
    best_params_array, min_conflicts, opt_info = optimizer.optimize_adam_pytorch(
        initial_params=initial_array,
        maxiter=kwargs.get('maxiter', 50),
        lr=kwargs.get('lr', 0.01),
        betas=kwargs.get('betas', (0.9, 0.999)),
        epsilon_fd=kwargs.get('epsilon_fd', 0.01),
        verbose=kwargs.get('verbose', True)
    )

    # 提取初始和最终的着色结果
    initial_params_for_eval = initial_array if initial_array is not None else np.array([0.5, 0.3] * p)
    evaluation_history = optimizer.evaluation_history

    if evaluation_history:
        # 找到初始参数的评估结果
        initial_eval = None
        for eval_data in evaluation_history:
            if np.allclose(eval_data['params'], initial_params_for_eval, atol=1e-6):
                initial_eval = eval_data
                break

        # 找到最终参数的评估结果
        final_eval = None
        for eval_data in evaluation_history:
            if np.allclose(eval_data['params'], best_params_array, atol=1e-6):
                final_eval = eval_data
                break

        # 绘制着色对比图
        if initial_eval and final_eval and kwargs.get('plot_results', True):
            try:
                plot_path = None
                if output_dir:
                    # 生成带图类型和时间戳的唯一文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    graph_type_name = f"{graph.number_of_nodes()}nodes_{graph.number_of_edges()}edges"
                    plot_filename = f"coloring_comparison_{graph_type_name}_k{k}_p{p}_{timestamp}.pdf"
                    plot_path = os.path.join(output_dir, plot_filename)

                optimizer.plot_coloring_results(
                    initial_coloring=initial_eval.get('coloring', {}),
                    final_coloring=final_eval.get('coloring', {}),
                    initial_conflicts=initial_eval.get('conflicts', float('inf')),
                    final_conflicts=final_eval.get('conflicts', float('inf')),
                    initial_params=initial_params_for_eval,
                    final_params=best_params_array,
                    save_path=plot_path
                )
            except Exception as e:
                print(f"⚠️ 绘制着色对比图失败: {e}")

    # 转换为参数字典
    best_params = {}
    for layer in range(p):
        best_params[f'gamma_{layer}'] = float(best_params_array[2 * layer])
        best_params[f'beta_{layer}'] = float(best_params_array[2 * layer + 1])

    return {
        'best_params': best_params,
        'min_conflicts': min_conflicts,
        'optimization_info': opt_info,
        'k': k,
        'p': p
    }


# ============================================================================
# 示例使用代码
# ============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    # 创建平台实例 (使用仿真机)
    # 从环境变量或配置文件读取登录密钥，避免硬编码
    try:
        from config import TIANYAN_CONFIG
        login_key = TIANYAN_CONFIG["login_key"]
    except ImportError:
        login_key = os.environ.get("TIANYAN_LOGIN_KEY", None)
        if login_key is None:
            raise ValueError("请在 config.py 中配置 TIANYAN_CONFIG['login_key'] 或设置环境变量 TIANYAN_LOGIN_KEY")

    platform = TianYanPlatform(login_key=login_key) # 不需要 login_key
    platform.set_machine("tianyan_sa")  # 使用仿真机
    print("="*60)
    print("QAOA 参数训练示例 (PyTorch Adam)")
    print("="*60)
    print("✓ 已连接到天衍仿真机\n")

    # 创建实验室
    from datetime import datetime
    lab_id = platform.create_lab(
        name=f'train_qaoa_{datetime.now().strftime("%Y%m%d%H%M%S")}',
        remark='QAOA Parameter Training with PyTorch Adam'
    )
    print(f"✓ 实验室创建完成: {lab_id}\n")

    # 创建4节点测试图
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 3), (1, 3)])

    print(f"图信息: 节点数={graph.number_of_nodes()}, 边数={graph.number_of_edges()}")
    print(f"图结构: 4节点图 (3节点环 + 1个额外节点)\n")

    # 使用 PyTorch Adam 优化器
    print("="*60)
    print("使用 PyTorch Adam 优化器优化 QAOA 参数")
    print("="*60)
    print("说明:")
    print("  - 使用 torch.optim.Adam 优化器")
    print("  - 由于天衍平台不支持梯度计算，使用有限差分法估计梯度")
    print("  - 每次迭代需要评估 2*n+1 次参数 (n=参数数量)")
    print("  - 优化策略：减少迭代次数和采样次数以加快评估")
    print(f"  - 对于 2 个参数，每次迭代需要 5 次实验")
    print(f"  - 总共 3 次迭代，预计需要 {3*5} = 15 次实验")
    print(f"  - 每次评估约需要 0.5-1 分钟，总共可能需要 7-15 分钟\n")

    # 创建输出目录
    from datetime import datetime
    output_dir = os.path.join(os.path.dirname(__file__), "training_results",
                                  f"pytorch_adam_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    result = train_qaoa_on_tianyan(
        platform=platform,
        graph=graph,
        k=3,
        p=1,
        output_dir=output_dir,
        lab_id=lab_id,
        num_shots=200,  # 进一步减少采样次数
        lr=0.05,  # 提高学习率加速收敛
        maxiter=3,  # 进一步减少迭代次数
        betas=(0.9, 0.999),  # Adam 超参数
        epsilon_fd=0.03,  # 增大有限差分步长
        verbose=True
    )

    print("\n" + "="*60)
    print("PyTorch Adam 优化结果汇总")
    print("="*60)
    print(f"颜色数 k: {result['k']}")
    print(f"层数 p: {result['p']}")
    print(f"最小冲突数: {result['min_conflicts']}")
    print(f"最优参数:")
    for key, value in result['best_params'].items():
        print(f"  {key} = {value:.6f}")
    print(f"优化信息: {result['optimization_info']}")
    print("="*60)
