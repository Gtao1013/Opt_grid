import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import norm, qmc


os.environ['OMP_NUM_THREADS'] = '1'

@dataclass
class OptimizationPhase:
    name: str
    iterations: tuple  # (start_iteration, end_iteration)
    criterion: str
    n_points: int
    balance_ratio: float


class SamplingStrategy:
    """加点采样策略基类"""

    def __init__(self, var_bounds, refinement_ratio=0.2, n_samples_per_point=3):
        self.var_bounds = var_bounds
        self.refinement_ratio = refinement_ratio
        self.n_samples_per_point = n_samples_per_point

    def generate_refinement_points(self, optimal_points: np.ndarray) -> np.ndarray:
        """在最优点周围生成加点"""
        if len(optimal_points) == 0:
            return np.array([])

        new_points = []
        var_ranges = self.var_bounds[:, 1] - self.var_bounds[:, 0]

        for point in optimal_points:
            # 在每个维度上加点
            for i in range(len(point)):
                delta = var_ranges[i] * self.refinement_ratio
                left_point = point.copy()
                right_point = point.copy()
                left_point[i] = max(point[i] - delta, self.var_bounds[i, 0])
                right_point[i] = min(point[i] + delta, self.var_bounds[i, 1])
                new_points.extend([left_point, right_point])

            # 在点周围添加随机扰动的点
            for _ in range(self.n_samples_per_point):
                random_point = point + np.random.uniform(
                    -self.refinement_ratio,
                    self.refinement_ratio,
                    size=len(point)
                ) * var_ranges
                random_point = np.clip(
                    random_point,
                    self.var_bounds[:, 0],
                    self.var_bounds[:, 1]
                )
                new_points.append(random_point)

        return np.array(new_points)


class AdaptiveSamplingStrategy:
    def __init__(self, var_bounds: np.ndarray, total_iterations: int = 40):
        self.var_bounds = var_bounds
        self.total_iterations = total_iterations
        self.exploration_ratio = 0.8

    def _calculate_mse(self, x, surrogate_models):
        """计算均方误差 (MSE) 准则"""
        try:
            mse_values = []
            for obj_name, model in surrogate_models.items():
                if hasattr(model, 'predict_std'):
                    # 使用predict_std方法
                    std = model.predict_std(x.reshape(1, -1))
                    mse_values.append(std[0] ** 2)
                else:
                    # 如果模型没有predict_std方法，使用默认值
                    print(f"警告: 模型 {obj_name} 缺少predict_std方法，使用默认方差")
                    mse_values.append(0.01)  # 使用小的默认方差

            if not mse_values:  # 如果列表为空
                return 0.01

            return np.mean(mse_values)
        except Exception as e:
            print(f"计算MSE时出错: {str(e)}")
            # 返回一个默认值以避免程序崩溃
            return 0.01

    def _calculate_pi(self, x, surrogate_models, y_best):
        """计算改进概率 (PI) 准则"""
        try:
            pi_values = []
            for obj_name, model in surrogate_models.items():
                y_pred = model.predict(x.reshape(1, -1))

                # 检查model是否有predict_std方法
                if hasattr(model, 'predict_std'):
                    std = model.predict_std(x.reshape(1, -1))
                else:
                    # 使用默认值
                    print(f"警告: 模型 {obj_name} 缺少predict_std方法，使用默认标准差")
                    std = np.array([0.1])

                # 避免除以零
                denominator = std + 1e-6
                z = (y_best - y_pred) / denominator
                pi = norm.cdf(z)
                pi_values.append(pi[0])

            if not pi_values:  # 如果列表为空
                return 0.5

            return np.mean(pi_values)
        except Exception as e:
            print(f"计算PI时出错: {str(e)}")
            return 0.5

    def _calculate_msp(self, x, surrogate_models):
        """计算代理模型预测最小值 (MSP) 准则"""
        try:
            predictions = []
            for obj_name, model in surrogate_models.items():
                y_pred = model.predict(x.reshape(1, -1))
                predictions.append(y_pred[0])

            if not predictions:  # 如果列表为空
                return 0

            return np.mean(predictions)
        except Exception as e:
            print(f"计算MSP时出错: {str(e)}")
            return 0

    def _calculate_lcb(self, x, surrogate_models, kappa=2.0):
        """计算置信下界 (LCB) 准则"""
        try:
            lcb_values = []
            for obj_name, model in surrogate_models.items():
                y_pred = model.predict(x.reshape(1, -1))

                # 检查model是否有predict_std方法
                if hasattr(model, 'predict_std'):
                    std = model.predict_std(x.reshape(1, -1))
                else:
                    # 使用默认值
                    print(f"警告: 模型 {obj_name} 缺少predict_std方法，使用默认标准差")
                    std = np.array([0.1])

                lcb = y_pred - kappa * std
                lcb_values.append(lcb[0])

            if not lcb_values:  # 如果列表为空
                return 0

            return np.mean(lcb_values)
        except Exception as e:
            print(f"计算LCB时出错: {str(e)}")
            return 0

    def _calculate_ei(self, x, surrogate_models, y_best):
        """计算期望改进 (EI) 准则"""
        try:
            ei_values = []
            for obj_name, model in surrogate_models.items():
                y_pred = model.predict(x.reshape(1, -1))

                # 检查model是否有predict_std方法
                if hasattr(model, 'predict_std'):
                    std = model.predict_std(x.reshape(1, -1))
                else:
                    # 使用默认值
                    print(f"警告: 模型 {obj_name} 缺少predict_std方法，使用默认标准差")
                    std = np.array([0.1])

                # 避免除以零
                denominator = std + 1e-6
                z = (y_best - y_pred) / denominator
                ei = (y_best - y_pred) * norm.cdf(z) + std * norm.pdf(z)
                ei_values.append(ei[0])

            if not ei_values:  # 如果列表为空
                return 0

            return np.mean(ei_values)
        except Exception as e:
            print(f"计算EI时出错: {str(e)}")
            return 0

    def _normalize_scores(self, scores):
        """归一化评分到 [0, 1] 范围"""
        if len(scores) == 0:
            return scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def _generate_explore_points(self, n_points, surrogate_models, iteration):
        """使用多准则融合的全局探索策略"""
        try:
            # 使用LHS生成候选点
            n_candidates = min(n_points * 20, 200)
            sampler = qmc.LatinHypercube(d=self.var_bounds.shape[0])
            candidates_normalized = sampler.random(n_candidates)
            candidates = qmc.scale(
                candidates_normalized,
                self.var_bounds[:, 0],
                self.var_bounds[:, 1]
            )

            # 计算当前最优值
            y_best = float('inf')
            for model in surrogate_models.values():
                y_pred = model.predict(candidates)
                y_best = min(y_best, np.min(y_pred))

            # 根据优化阶段调整准则权重
            progress = iteration / self.total_iterations
            weights = self._get_adaptive_weights(progress)

            # 计算每个候选点的各项准则得分
            scores = self._calculate_multi_criteria_scores(
                candidates, surrogate_models, y_best, weights
            )

            # 选择得分最高的点
            best_indices = np.argsort(scores)[-n_points:]
            selected_points = candidates[best_indices]

            # 添加局部扰动以增加多样性
            selected_points = self._add_local_perturbation(selected_points)

            return selected_points

        except Exception as e:
            print(f"生成探索点时出错: {str(e)}")
            raise

    def _generate_exploit_points(self, pareto_solutions, surrogate_models, n_points):
        """生成利用点"""
        if len(pareto_solutions) == 0:
            return []

        try:
            # 使用代理模型评估帕累托解的预测值和不确定性
            predictions = []
            uncertainties = []
            for solution in pareto_solutions:
                pred_values = []
                uncert_values = []
                for obj_name, model in surrogate_models.items():
                    pred = model.predict(solution.reshape(1, -1))
                    pred_values.append(pred[0])

                    # 检查model是否有predict_std方法
                    if hasattr(model, 'predict_std'):
                        std = model.predict_std(solution.reshape(1, -1))
                        uncert_values.append(std[0])
                    else:
                        # 使用默认值
                        print(f"警告: 模型 {obj_name} 缺少predict_std方法，使用默认标准差")
                        uncert_values.append(0.1)

                predictions.append(np.mean(pred_values))
                uncertainties.append(np.mean(uncert_values))

            # 选择预测值最好且不确定性较高的点周围进行采样
            # 使用综合得分来选择基准点
            scores = np.array(predictions) - 0.5 * np.array(uncertainties)  # 平衡探索和利用
            best_indices = np.argsort(scores)[:n_points]

            points = []
            for idx in best_indices:
                base_point = pareto_solutions[idx]

                # 在最优点附近添加局部搜索点
                for _ in range(3):  # 为每个基准点生成多个候选点
                    # 自适应扰动大小：基于不确定性
                    perturbation_scale = 0.1 * (1 + uncertainties[idx])
                    perturbation = np.random.normal(
                        0,
                        perturbation_scale,
                        size=len(base_point)
                    )

                    # 生成新点
                    new_point = base_point + perturbation * (self.var_bounds[:, 1] - self.var_bounds[:, 0])

                    # 确保在边界内
                    new_point = np.clip(
                        new_point,
                        self.var_bounds[:, 0],
                        self.var_bounds[:, 1]
                    )

                    points.append(new_point)

            # 如果生成了足够多的候选点，使用聚类选择最终的点
            if len(points) > n_points:
                points = np.array(points)
                kmeans = KMeans(n_clusters=n_points, random_state=42)
                cluster_labels = kmeans.fit_predict(points)

                # 从每个簇中选择最接近中心的点
                final_points = []
                for i in range(n_points):
                    cluster_points = points[cluster_labels == i]
                    if len(cluster_points) > 0:
                        cluster_center = kmeans.cluster_centers_[i]
                        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                        best_point_idx = np.argmin(distances)
                        final_points.append(cluster_points[best_point_idx])

                points = np.array(final_points)
            else:
                points = np.array(points[:n_points])

            # 确保返回足够的点
            if len(points) < n_points:
                # 如果点不够，添加随机点补充
                n_additional = n_points - len(points)
                additional_points = np.random.uniform(
                    self.var_bounds[:, 0],
                    self.var_bounds[:, 1],
                    size=(n_additional, len(self.var_bounds))
                )
                points = np.vstack([points, additional_points]) if len(points) > 0 else additional_points

            print(f"\n生成了 {len(points)} 个利用点")
            print(f"利用点范围:")
            for i, (lower, upper) in enumerate(self.var_bounds):
                points_min = points[:, i].min()
                points_max = points[:, i].max()
                print(f"维度 {i}: [{points_min:.4f}, {points_max:.4f}]")

            return points

        except Exception as e:
            print(f"生成利用点时出错: {str(e)}")
            raise

    def _get_adaptive_weights(self, progress):
        """根据优化进度返回自适应权重"""
        if progress < 0.3:  # 早期阶段
            return {
                'mse': 0.4, 'pi': 0.2, 'lcb': 0.2,
                'msp': 0.1, 'ei': 0.1
            }
        elif progress < 0.7:  # 中期阶段
            return {
                'mse': 0.2, 'pi': 0.2, 'lcb': 0.2,
                'msp': 0.2, 'ei': 0.2
            }
        else:  # 后期阶段
            return {
                'mse': 0.1, 'pi': 0.2, 'lcb': 0.2,
                'msp': 0.3, 'ei': 0.2
            }

    def _calculate_multi_criteria_scores(self, candidates, surrogate_models, y_best, weights):
        """计算多准则加权得分"""
        scores = np.zeros(len(candidates))

        for i, x in enumerate(candidates):
            criteria_scores = {}

            # 尝试计算各项准则得分
            try:
                criteria_scores['mse'] = self._calculate_mse(x, surrogate_models)
            except Exception as e:
                print(f"计算MSE得分出错: {str(e)}")
                criteria_scores['mse'] = 0

            try:
                criteria_scores['pi'] = self._calculate_pi(x, surrogate_models, y_best)
            except Exception as e:
                print(f"计算PI得分出错: {str(e)}")
                criteria_scores['pi'] = 0

            try:
                criteria_scores['lcb'] = self._calculate_lcb(x, surrogate_models)
            except Exception as e:
                print(f"计算LCB得分出错: {str(e)}")
                criteria_scores['lcb'] = 0

            try:
                criteria_scores['msp'] = self._calculate_msp(x, surrogate_models)
            except Exception as e:
                print(f"计算MSP得分出错: {str(e)}")
                criteria_scores['msp'] = 0

            try:
                criteria_scores['ei'] = self._calculate_ei(x, surrogate_models, y_best)
            except Exception as e:
                print(f"计算EI得分出错: {str(e)}")
                criteria_scores['ei'] = 0

            # 归一化各项得分
            normalized_scores = {}
            for criterion, score in criteria_scores.items():
                normalized_scores[criterion] = self._normalize_scores([score])[0]

            # 组合加权得分
            combined_score = sum(weights.get(criterion, 0) * score
                                 for criterion, score in normalized_scores.items())
            scores[i] = combined_score

        return scores

    def _add_local_perturbation(self, points):
        """添加局部扰动以增加多样性"""
        noise = np.random.normal(
            0,
            0.05,
            size=points.shape
        ) * (self.var_bounds[:, 1] - self.var_bounds[:, 0])

        points = np.clip(
            points + noise,
            self.var_bounds[:, 0],
            self.var_bounds[:, 1]
        )
        return points

    def select_points(self, population, objectives, pareto_front_indices,
                      surrogate_models, iteration, n_points=3):
        """选择新的采样点"""
        try:
            # 根据迭代进度调整探索比例
            progress = iteration / self.total_iterations
            self.exploration_ratio = max(0.2, 0.8 - 0.6 * progress)

            n_explore = int(n_points * self.exploration_ratio)
            n_exploit = n_points - n_explore

            selected_points = []

            # 探索点（使用多准则融合策略）
            if n_explore > 0:
                explore_points = self._generate_explore_points(
                    n_explore,
                    surrogate_models,
                    iteration
                )
                selected_points.extend(explore_points)

            # 利用点（局部搜索）
            if n_exploit > 0 and len(pareto_front_indices) > 0:
                pareto_solutions = population[pareto_front_indices]
                exploit_points = self._generate_exploit_points(
                    pareto_solutions,
                    surrogate_models,
                    n_exploit
                )
                selected_points.extend(exploit_points)

            # 确保所有点在边界内并去重
            selected_points = np.array(selected_points)
            selected_points = np.clip(
                selected_points,
                self.var_bounds[:, 0],
                self.var_bounds[:, 1]
            )
            selected_points = np.unique(selected_points, axis=0)

            print(f"\n选择了 {len(selected_points)} 个新采样点:")
            print(f"探索点数: {n_explore}")
            print(f"利用点数: {n_exploit}")
            print(f"当前探索比例: {self.exploration_ratio:.2f}")

            return selected_points

        except Exception as e:
            print(f"采样点选择出错: {str(e)}")
            raise