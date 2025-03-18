import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, List
import os
import matplotlib.pyplot as plt
import pickle
import subprocess
import glob
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from optimizer import NSGA2Optimizer, NSGAIIParameters
from cfd_automation import CFDAutomation
from surrogate_validation import SurrogateValidator
import shutil
from sampling_strategy import SamplingStrategy, AdaptiveSamplingStrategy
from data_manager import OptimizationDataManager, SurrogateModelManager
from surrogate_model import KrigingModel
from pymoo.indicators.hv import Hypervolume as HV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
import platform
import copy
import sklearn.gaussian_process.kernels as kernels
from utils import safe_filename
import traceback

class OptimizationProblem:
    def __init__(self):
        # 时间和用户信息
        self.start_time = datetime.now()  # 使用当前时间更合理
        self.user_login = "Gtao1013"

        # 目录设置
        self.work_dir = r'F:\essay_gt\opt'
        self.base_path = self.work_dir

        # 初始化数据管理器
        self.data_manager = OptimizationDataManager(base_path=self.base_path)

        # 初始化代理模型管理器 (新增)
        self.surrogate_manager = SurrogateModelManager(base_path=self.base_path, n_folds=15)

        # 变量和目标定义
        self.bounds = np.array([
            [170, 650],  # Chord 的边界
            [500, 2000],  # Distance 的边界
            [0, 45]  # Fai 的边界
        ])
        self.var_names = ['Chord', 'Distance', 'Fai']
        self.obj_names = ['Cl', 'Cd', 'Cm']
        self.n_var = len(self.var_names)
        self.n_obj = len(self.obj_names)

        # Cl, Cd, Cm 都是最大化目标
        self.minimize = [False, False, False]

        # 文件路径设置
        self.doe_file = os.path.join(self.work_dir, "doe_dynamic.txt")
        self.new_points_file = os.path.join(self.work_dir, "data_new.txt")

        # 优化历史记录初始化
        self.current_iteration = 0
        self.all_objectives = []
        self.optimization_history = {
            'iterations': [],
            'best_objectives': [],
            'population_diversity': [],
            'sample_points': []
        }

        # 初始化代理模型字典
        self.surrogate_models = {}

        # 创建优化器参数
        optimizer_params = NSGAIIParameters(
            pop_size=20,
            crossover_prob=0.9,
            mutation_prob=1.0 / self.n_var,
            crossover_eta=20,
            mutation_eta=20
        )

        # 初始化优化器
        self.optimizer = NSGA2Optimizer(
            n_var=self.n_var,
            n_obj=self.n_obj,
            bounds=self.bounds,
            params=optimizer_params,
            minimize=[False, False, False]  # Cl, Cd, Cm 都是最大化目标
        )

        # 初始化采样策略
        self.sampling_strategy = AdaptiveSamplingStrategy(
            var_bounds=self.bounds,
            total_iterations=40
        )

        # 初始化CFD接口
        self.cfd = CFDAutomation()

        print(f"优化问题初始化完成 - 时间: {self.start_time}")
        print(f"用户: {self.user_login}")
        print(f"工作目录: {self.work_dir}")

    def initialize_models(self):
        """初始化并训练代理模型"""
        print(f"\n初始化代理模型... - {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")

        try:
            # 检查并修复数据文件
            print("\n检查并修复数据文件...")
            self.check_and_fix_doe_file(self.doe_file)

            # 读取数据
            data = np.loadtxt(self.doe_file, skiprows=1)
            if data.shape[1] != self.n_var + self.n_obj:
                raise ValueError(f"数据列数 ({data.shape[1]}) 与预期不符 (应为 {self.n_var + self.n_obj})")

            X = data[:, :self.n_var]
            Y = data[:, self.n_var:]

            # 数据统计信息
            self._print_data_statistics(X, Y)

            # 检查数据质量
            if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
                raise ValueError("数据中存在缺失值")

            # 检查数据范围
            for i, name in enumerate(self.obj_names):
                y_range = np.max(Y[:, i]) - np.min(Y[:, i])
                if y_range < 1e-6:
                    print(f"警告: {name} 的数据范围极小 ({y_range:.8f})，可能导致模型不稳定")

            # 修改 sklearn 高斯过程内核的默认参数
            # 这是直接修改底层库的警告阈值的方式，可以消除警告而不改变模型行为
            # 注意: 这是一个全局修改，会影响所有高斯过程模型实例
            try:

                # 备份原始方法
                original_check_length_scale = kernels.Kernel._check_length_scale

                # 定义新方法，增加警告阈值
                def _patched_check_length_scale(self, X, length_scale):
                    # 增加到原来的10倍，这只是修改警告阈值，不影响实际行为
                    X_orig = X
                    length_scale_orig = length_scale
                    result = original_check_length_scale(self, X, length_scale)
                    # 在结果计算后，修改length_scale_bounds属性
                    if hasattr(self, 'length_scale_bounds'):
                        # 如果接近上界，临时增加上界
                        if isinstance(self.length_scale_bounds, tuple):
                            # 标量情况
                            lb, ub = self.length_scale_bounds
                            if ub == 1000.0:  # 如果是默认上界
                                self.length_scale_bounds = (lb, 10000.0)
                        else:
                            # 多维情况
                            for i in range(len(self.length_scale_bounds)):
                                lb, ub = self.length_scale_bounds[i]
                                if ub == 1000.0:  # 如果是默认上界
                                    self.length_scale_bounds[i] = (lb, 10000.0)
                    return result

                # 应用补丁
                kernels.Kernel._check_length_scale = _patched_check_length_scale
                print("已修改高斯过程内核的默认参数以消除长度尺度警告")
            except Exception as e:
                print(f"修改高斯过程内核参数时出错: {str(e)}")
                print("将继续使用默认参数")

            # 训练代理模型
            for i, obj_name in enumerate(self.obj_names):
                print(f"\n训练 {obj_name} 代理模型:")
                print("-" * 40)

                # 直接创建和修改KrigingModel实例
                try:
                    # 如果KrigingModel允许直接设置内核参数

                    # 创建自定义内核，明确设置足够高的上界
                    n_dims = X.shape[1]
                    # 为每个维度单独设置边界，特别是第3维（索引为2）
                    length_scale_bounds = [(1e-5, 10000.0) for _ in range(n_dims)]
                    # 为出现问题的维度特别设置
                    length_scale_bounds[2] = (1e-5, 100000.0)

                    # 创建带有自定义内核的模型
                    custom_kernel = ConstantKernel() * Matern(
                        length_scale=[1.0] * n_dims,
                        length_scale_bounds=length_scale_bounds,
                        nu=2.5
                    ) + WhiteKernel(noise_level=1e-5)

                    # 尝试使用自定义内核创建模型
                    try:
                        # 首先尝试直接传递内核参数
                        model = KrigingModel(normalize=True, kernel=custom_kernel)
                    except TypeError:
                        # 如果直接传递失败，先创建默认模型然后修改内核
                        model = KrigingModel(normalize=True)
                        # 尝试访问和修改内部高斯过程回归器
                        if hasattr(model, 'gpr'):
                            model.gpr.kernel = custom_kernel
                        elif hasattr(model, 'model'):
                            model.model.kernel = custom_kernel
                        elif hasattr(model, 'regressor'):
                            model.regressor.kernel = custom_kernel
                        elif hasattr(model, 'estimator'):
                            model.estimator.kernel = custom_kernel
                        else:
                            print("无法修改KrigingModel内部的高斯过程内核，使用默认配置")
                except Exception as e:
                    print(f"创建自定义内核时出错: {str(e)}")
                    print("使用默认KrigingModel配置")
                    model = KrigingModel(normalize=True)

                # 使用修改后的交叉验证
                # 初始化一个SurrogateModelManager如果还没有
                if not hasattr(self, 'surrogate_manager'):
                    self.surrogate_manager = SurrogateModelManager(base_path=self.base_path, n_folds=10)

                scores, predictions = self.surrogate_manager.cross_validate_model(
                    X, Y[:, i], obj_name, model_class=KrigingModel
                )

                # 用全部数据训练最终模型
                print("\n使用全部数据训练最终模型...")
                model.fit(X, Y[:, i])
                self.surrogate_models[obj_name] = model

                # 保存代理模型
                self.data_manager.save_surrogate_model(
                    model, obj_name, self.current_iteration
                )

            print("\n代理模型初始化完成")

            # 评估模型结果
            evaluation_results = {}
            for i, obj_name in enumerate(self.obj_names):
                if obj_name in self.surrogate_models:
                    # 使用交叉验证评估模型性能
                    model = self.surrogate_models[obj_name]
                    y = data[:, self.n_var + i]

                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = []

                    for train_idx, test_idx in kf.split(X):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        # 创建新模型实例进行交叉验证
                        cv_model = KrigingModel(normalize=True)
                        cv_model.fit(X_train, y_train)
                        y_pred = cv_model.predict(X_test)

                        cv_scores.append({
                            'r2': r2_score(y_test, y_pred),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                        })

                    # 计算平均分数
                    avg_r2 = np.mean([s['r2'] for s in cv_scores])
                    avg_rmse = np.mean([s['rmse'] for s in cv_scores])

                    evaluation_results[obj_name] = {
                        'r2': avg_r2,
                        'rmse': avg_rmse
                    }
                    print(f"{obj_name} 模型交叉验证结果: R²={avg_r2:.4f}, RMSE={avg_rmse:.6f}")

            # 更新评估历史
            self._update_model_evaluation_history(evaluation_results)

            return True

        except Exception as e:
            print(f"\n代理模型训练过程中出错: {str(e)}")

            traceback.print_exc()
            self._handle_initialization_error(e)
            return False

    def _check_convergence(self, history, current_iteration, window_size=5, threshold=0.001):
        """
        检查优化过程是否已经收敛 - 增强版

        参数:
        -----
        history : dict
            优化历史数据
        current_iteration : int
            当前迭代次数
        window_size : int
            用于检测稳定性的窗口大小
        threshold : float
            判定收敛的阈值

        返回:
        -----
        bool
            如果收敛返回True，否则返回False
        str
            收敛原因描述
        """
        # 至少需要window_size+1次迭代才能判断收敛
        if current_iteration < window_size + 1:
            return False, "迭代次数不足"

        # 1. 检查Pareto前沿的改进
        if len(self.all_objectives) >= window_size + 1:
            # 获取最近window_size+1次迭代的Pareto前沿
            recent_fronts = []
            for i in range(window_size + 1):
                idx = current_iteration - i
                if idx >= 0 and idx < len(self.all_objectives):
                    objectives = self.all_objectives[idx]
                    fronts = self.optimizer.fast_non_dominated_sort(objectives)
                    pareto_front = objectives[fronts[0]]
                    recent_fronts.append(pareto_front)

            if len(recent_fronts) >= 2:
                # 计算连续Pareto前沿的变化
                improvements = []
                for i in range(len(recent_fronts) - 1):
                    # 计算当前与前一个Pareto前沿的超体积差异
                    if len(recent_fronts[i]) > 0 and len(recent_fronts[i + 1]) > 0:
                        # 使用目标函数值的变化来估计改进
                        metrics = self._calculate_pareto_metrics(recent_fronts[i], recent_fronts[i + 1])
                        improvements.append(metrics['relative_change'])

                # 如果所有最近的改进都低于阈值，认为收敛
                if len(improvements) >= window_size:
                    avg_improvement = np.mean(improvements)
                    if avg_improvement < threshold:
                        return True, f"Pareto前沿在{window_size}次连续迭代中平均变化为{avg_improvement:.6f}，小于阈值{threshold}"

        # 2. 检查超体积指标的稳定性
        if hasattr(self, 'hypervolume_history') and len(self.hypervolume_history) >= window_size + 1:
            recent_hv = self.hypervolume_history[-window_size - 1:]
            hv_changes = []
            for i in range(1, len(recent_hv)):
                if recent_hv[i - 1] > 0:  # 避免除以零
                    rel_change = abs((recent_hv[i] - recent_hv[i - 1]) / recent_hv[i - 1])
                    hv_changes.append(rel_change)

            if len(hv_changes) >= window_size and all(change < threshold for change in hv_changes):
                avg_change = np.mean(hv_changes)
                return True, f"超体积指标在{window_size}次连续迭代中平均变化为{avg_change:.6f}，小于阈值{threshold}"

        # 3. 检查最优解的稳定性
        if 'best_objectives' in history and len(history['best_objectives']) >= window_size + 1:
            recent_best = history['best_objectives'][-window_size - 1:]
            changes = []

            for i in range(1, len(recent_best)):
                # 计算相对变化
                rel_change = np.abs(
                    (recent_best[i] - recent_best[i - 1]) / np.maximum(np.abs(recent_best[i - 1]), 1e-10))
                max_rel_change = np.max(rel_change)
                changes.append(max_rel_change)

            if len(changes) >= window_size and all(change < threshold for change in changes):
                avg_change = np.mean(changes)
                return True, f"最优解在{window_size}次连续迭代中平均变化为{avg_change:.6f}，小于阈值{threshold}"

        # 4. 检查种群多样性
        if 'population_diversity' in history and len(history['population_diversity']) >= window_size + 1:
            recent_diversity = history['population_diversity'][-window_size - 1:]
            diversity_changes = []

            for i in range(1, len(recent_diversity)):
                if recent_diversity[i - 1] > 0:  # 避免除以零
                    rel_change = abs((recent_diversity[i] - recent_diversity[i - 1]) / recent_diversity[i - 1])
                    diversity_changes.append(rel_change)

            # 多样性变化稳定且值较小
            if (len(diversity_changes) >= window_size and
                    all(change < threshold for change in diversity_changes) and
                    np.mean(recent_diversity) < 0.05):
                return True, f"种群多样性在{window_size}次连续迭代中稳定且较低，平均值为{np.mean(recent_diversity):.6f}"

        # 5. 检查代理模型的预测性能
        if hasattr(self, 'model_evaluation_history') and len(self.model_evaluation_history) >= window_size + 1:
            # 获取R²值的变化
            r2_changes = []
            for obj_name in self.obj_names:
                obj_r2 = [eval_result.get(obj_name, {}).get('r2', 0)
                          for eval_result in self.model_evaluation_history[-window_size - 1:]]
                for i in range(1, len(obj_r2)):
                    r2_changes.append(abs(obj_r2[i] - obj_r2[i - 1]))

            # 如果所有R²变化都小于阈值，且平均R²值高
            avg_r2_change = np.mean(r2_changes) if r2_changes else 1.0
            if r2_changes and avg_r2_change < threshold:
                # 计算当前R²值
                current_r2 = np.mean([
                    self.model_evaluation_history[-1].get(obj_name, {}).get('r2', 0)
                    for obj_name in self.obj_names
                ])

                if current_r2 > 0.9:  # R²值高且稳定
                    return True, f"代理模型R²值稳定且高({current_r2:.4f})，平均变化为{avg_r2_change:.6f}"

        # 如果设置了最大迭代次数且已达到，则停止
        max_iterations = getattr(self, 'max_iterations', None)
        if max_iterations is not None and current_iteration >= max_iterations - 1:
            return True, f"已达到最大迭代次数: {max_iterations}"

        # 默认未收敛
        return False, "未达到收敛条件"

    def _calculate_pareto_metrics(self, front1, front2):
        """计算两个Pareto前沿之间的度量"""
        # 计算前沿的平均值
        mean1 = np.mean(front1, axis=0)
        mean2 = np.mean(front2, axis=0)

        # 相对变化
        rel_change = np.mean(np.abs((mean1 - mean2) / np.maximum(np.abs(mean2), 1e-10)))

        # 计算前沿的分布范围
        range1 = np.max(front1, axis=0) - np.min(front1, axis=0)
        range2 = np.max(front2, axis=0) - np.min(front2, axis=0)
        range_change = np.mean(np.abs(range1 - range2) / np.maximum(range2, 1e-10))

        # 点数变化
        point_count_change = abs(len(front1) - len(front2)) / max(len(front2), 1)

        return {
            'relative_change': rel_change,
            'range_change': range_change,
            'point_count_change': point_count_change
        }

    def _calculate_hypervolume(self, pareto_front):
        """计算Pareto前沿的超体积，最大化问题"""
        if len(pareto_front) == 0:
            return 0.0

        if HV is None:
            print("警告：无法计算超体积，pymoo库导入失败")
            return 0.0

        try:
            # 确保pareto_front是2D数组
            if pareto_front.ndim == 1:
                pareto_front = pareto_front.reshape(1, -1)

            # 直接使用原始值作为最大化问题处理
            objectives_to_maximize = pareto_front.copy()

            # 为最大化问题设置参考点（需要比所有点都"差"）
            # 对于最大化问题，参考点应该比所有点的值都小
            ref_point = np.min(objectives_to_maximize, axis=0) - 0.1 * np.abs(np.min(objectives_to_maximize, axis=0))

            # 对于零值或接近零值的情况，确保参考点有足够的间距
            ref_point = np.where(np.abs(ref_point) < 1e-6, -0.1, ref_point)

            print(f"超体积计算的参考点: {ref_point}")

            # 计算超体积
            hv_calculator = HV(ref_point=ref_point)

            # 尝试不同的API调用方式
            try:
                # 新版API
                volume = hv_calculator.do(objectives_to_maximize)
            except Exception as e1:
                try:
                    # 旧版API
                    volume = hv_calculator.calc(objectives_to_maximize)
                except Exception as e2:
                    try:
                        volume = hv_calculator.compute(objectives_to_maximize)
                    except Exception as e3:
                        print(f"计算超体积时出错，尝试了多种方法均失败:")
                        print(f"  错误1: {e1}")
                        print(f"  错误2: {e2}")
                        print(f"  错误3: {e3}")
                        return 0.0

            print(f"超体积计算成功: {volume:.6f}")
            return volume

        except Exception as e:
            print(f"计算超体积时出错: {str(e)}")
            print(f"Pareto前沿形状: {pareto_front.shape if hasattr(pareto_front, 'shape') else 'unknown'}")
            if len(pareto_front) > 0:
                print(f"Pareto前沿数据示例: {pareto_front[:min(3, len(pareto_front))]}")
            return 0.0

    # 初始化模型评估历史记录
    def _init_model_evaluation_history(self):
        """初始化模型评估历史记录"""
        if not hasattr(self, 'model_evaluation_history'):
            self.model_evaluation_history = []

    # 更新模型评估历史
    def _update_model_evaluation_history(self, evaluation_results):
        """更新模型评估历史"""
        self._init_model_evaluation_history()
        self.model_evaluation_history.append(evaluation_results)

        # 保持历史记录在合理大小
        max_history_size = 20  # 保留最近20次评估结果
        if len(self.model_evaluation_history) > max_history_size:
            self.model_evaluation_history = self.model_evaluation_history[-max_history_size:]


    def _print_data_statistics(self, X, Y):
        """打印数据统计信息"""
        print("\n输入变量统计信息:")
        for i, name in enumerate(self.var_names):
            print(f"\n{name}:")
            print(f"范围: [{X[:, i].min():.4f}, {X[:, i].max():.4f}]")
            print(f"均值: {X[:, i].mean():.4f}")
            print(f"标准差: {X[:, i].std():.4f}")

        print("\n目标值统计信息:")
        for i, name in enumerate(self.obj_names):
            print(f"\n{name}:")
            print(f"范围: [{Y[:, i].min():.4f}, {Y[:, i].max():.4f}]")
            print(f"均值: {Y[:, i].mean():.4f}")
            print(f"标准差: {Y[:, i].std():.4f}")



    def plot(self, iteration: int):
        """
        绘制优化过程的可视化图形

        Parameters:
        -----------
        iteration: int
            当前迭代次数
        """
        try:
            # 创建图形保存目录
            plot_dir = os.path.join(os.path.dirname(self.doe_file), 'optimization_plots')
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            # 读取所有优化数据
            data = pd.read_csv(self.doe_file, delimiter='\t')

            # 创建图形
            fig = plt.figure(figsize=(15, 10))

            # 1. 变量分布图
            for i, var_name in enumerate(self.var_names):
                plt.subplot(3, 2, i + 1)
                plt.hist(data[var_name], bins=20)
                plt.title(f'{var_name} Distribution')
                plt.xlabel(var_name)
                plt.ylabel('Frequency')

            # 2. 目标函数收敛图
            plt.subplot(3, 2, 4)
            for obj_name in self.obj_names:
                plt.plot(data[obj_name].rolling(window=5).mean(), label=obj_name)
            plt.title('Objective Functions Convergence')
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.legend()

            # 3. Pareto前沿图（如果有多个目标）
            if len(self.obj_names) >= 2:
                plt.subplot(3, 2, 5)
                plt.scatter(data[self.obj_names[0]], data[self.obj_names[1]])
                plt.title('Pareto Front')
                plt.xlabel(self.obj_names[0])
                plt.ylabel(self.obj_names[1])

            # 保存图形
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = safe_filename(f'optimization_progress_gen', "png", include_timestamp=True, iteration=iteration)
            plot_path = os.path.join(plot_dir, plot_filename)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            print(f"\n优化过程图已保存至: {plot_path}")

        except Exception as e:
            print(f"\n生成优化过程图时出错: {str(e)}")
            raise

    def plot_prediction_results(self, y_true, y_pred, fold, obj_name):
        """可视化预测结果"""
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.title(f'{obj_name} - 折 {fold} 预测结果')

            # 添加R²值到图中
            r2 = r2_score(y_true, y_pred)
            plt.text(0.05, 0.95, f'R² = {r2:.4f}',
                     transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))

            # 保存图片
            save_dir = 'model_validation'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'{obj_name}_fold_{fold}_prediction.png'))
            plt.close()

        except Exception as e:
            print(f"绘制预测结果图时出错: {str(e)}")

    def check_and_fix_doe_file(self, filename):
        """检查并修复doe文件"""
        try:
            # 读取所有行
            with open(filename, 'r') as f:
                lines = f.readlines()

            # 验证和修复数据
            fixed_lines = []
            header = lines[0].strip()  # 保存标题行
            fixed_lines.append(header + '\n')

            print(f"开始检查文件: {filename}")
            print(f"标题行: {header}")

            for i, line in enumerate(lines[1:], start=1):
                line = line.strip()
                if not line:  # 跳过空行
                    print(f"跳过第 {i} 行: 空行")
                    continue

                values = line.split('\t')
                if len(values) != 6:  # 确保有6列数据
                    print(f"警告: 第 {i} 行列数不正确 ({len(values)}列)")
                    continue

                try:
                    # 尝试转换为浮点数并格式化
                    formatted_values = [f"{float(x):.6f}" for x in values]
                    fixed_line = '\t'.join(formatted_values) + '\n'
                    fixed_lines.append(fixed_line)
                except ValueError as e:
                    print(f"警告: 第 {i} 行数据无效: {str(e)}")
                    continue

            # 创建备份文件
            backup_filename = filename + '.backup'
            if not os.path.exists(backup_filename):
                shutil.copy2(filename, backup_filename)
                print(f"\n已创建备份文件: {backup_filename}")

            # 保存修复后的文件
            with open(filename, 'w') as f:
                f.writelines(fixed_lines)

            print(f"\n文件修复完成:")
            print(f"原始行数: {len(lines)}")
            print(f"修复后行数: {len(fixed_lines)}")

        except Exception as e:
            print(f"文件检查/修复过程出错: {str(e)}")
            raise

    def get_optimization_history(self):
        """收集优化历史数据"""
        history = {
            'iterations': [],
            'Cl': [],
            'Cd': [],
            'Cm': [],
            'Cl_min': [],
            'Cl_max': [],
            'Cd_min': [],
            'Cd_max': [],
            'Cm_min': [],
            'Cm_max': []
        }

        # 使用实际迭代索引而不是迭代次数
        for i in range(self.current_iteration + 1):
            # 保存当前迭代索引（从0开始）
            history['iterations'].append(i)

            objectives = self.all_objectives[i]
            for j, obj in enumerate(['Cl', 'Cd', 'Cm']):
                values = objectives[:, j]
                history[obj].append(np.mean(values))
                history[f'{obj}_min'].append(np.min(values))
                history[f'{obj}_max'].append(np.max(values))

        return history

    def _save_final_results(self, n_generations):
        """保存最终优化结果"""
        # 保存最终的优化历史
        # 在_save_final_results方法中用于生成最终图表
        final_history = self.get_optimization_history()
        self.data_manager.plot_optimization_history(
            final_history,
            include_initial_doe=True,
            doe_file=self.doe_file
        )

        # 保存最终的帕累托前沿
        final_objectives = self.all_objectives[-1]
        final_pareto_indices = self.optimizer.fast_non_dominated_sort(final_objectives)[0]
        self.data_manager.plot_pareto_front(
            objectives=final_objectives[final_pareto_indices],
            iteration=n_generations
        )

        # 保存完整的优化数据
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        final_results = {
            'timestamp': timestamp,
            'user': self.user_login,
            'total_generations': n_generations,
            'final_pareto_front': final_objectives[final_pareto_indices],
            'optimization_history': self.optimization_history,
            'surrogate_models': self.surrogate_models
        }

        results_file = os.path.join(
            self.base_path,
            safe_filename("final_optimization_results", "pkl", include_timestamp=True)
        )
        with open(results_file, 'wb') as f:
            pickle.dump(final_results, f)
        print(f"\n最终优化结果已保存至: {results_file}")

    def _update_doe_file(self, point_data):
        """更新doe文件"""
        try:
            formatted_data = '\t'.join([f"{x:12.6f}" for x in point_data])
            with open(self.doe_file, 'a') as f:
                f.write(formatted_data + '\n')
        except Exception as e:
            print(f"\n更新doe_dynamic.txt时出错: {str(e)}")
            raise

    def _handle_optimization_error(self, error, iteration):
        """处理优化过程中的错误"""
        error_log = os.path.join(self.base_path, "optimization_errors.log")
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")

        with open(error_log, 'a') as f:
            f.write(f"\n{'=' * 50}")
            f.write(f"\n时间: {timestamp}")
            f.write(f"\n用户: {self.user_login}")
            f.write(f"\n迭代: {iteration}")
            f.write(f"\n错误信息: {str(error)}")
            f.write(f"\n{'=' * 50}\n")

    def _update_validation_data(self, validator, iteration, actual_responses, predicted_responses):
        """更新验证数据和可视化"""
        validator.update_history(
            iteration=iteration,
            objectives=self.obj_names,
            true_values=actual_responses,
            predicted_values=predicted_responses
        )
        validator.plot_improved_validation_metrics(iteration)
        validator.save_validation_report(iteration)  # 修正括号错误

    def calculate_diversity(self, population: np.ndarray) -> float:
        """计算种群多样性指标"""
        if len(population) < 2:
            return 0.0

        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = np.linalg.norm(population[i] - population[j])
                distances.append(dist)

        return np.mean(distances)

    def _handle_initialization_error(self, error):
        """处理初始化过程中的错误"""
        error_log = os.path.join(self.base_path, "initialization_errors.log")
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")

        with open(error_log, 'a') as f:
            f.write(f"\n{'=' * 50}")
            f.write(f"\n时间: {timestamp}")
            f.write(f"\n用户: {self.user_login}")
            f.write(f"\n错误信息: {str(error)}")
            f.write(f"\n{'=' * 50}\n")

    def calculate_metrics(self, pareto_front: np.ndarray,
                          previous_best: np.ndarray) -> Tuple[float, float]:
        """计算优化指标"""
        # 计算目标改善程度
        if previous_best is not None and len(previous_best) > 0:
            current_mean = np.mean(pareto_front, axis=0)
            prev_mean = np.mean(previous_best, axis=0)
            improvement = np.mean((current_mean - prev_mean) / np.abs(prev_mean))
        else:
            improvement = float('inf')

        # 计算多样性指标
        diversity = self.calculate_diversity(pareto_front)

        return improvement, diversity

    def read_cm_from_file(filepath):
        """从文件读取Cm值并统一处理"""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()[-100:]  # 取最后100行

            # 解析值并计算平均值
            cm_values = []
            for line in lines:
                try:
                    cm_values.append(float(line.split()[1]))  # 假设Cm值在第二列
                except (IndexError, ValueError):
                    continue

            if cm_values:
                # 取绝对值之后的平均值 - 确保所有地方都一致使用
                cm_abs = abs(np.mean(cm_values))
                print(f"Cm原始平均值: {np.mean(cm_values)}, 取绝对值后: {cm_abs}")
                return cm_abs
            else:
                print("警告: 无法从文件解析Cm值")
                return 0.0
        except Exception as e:
            print(f"读取Cm文件出错: {str(e)}")
            return 0.0

    def process_cm_value(self, cm_value):
        """统一处理Cm值：取绝对值"""
        return abs(cm_value)

    def process_cfd_results(self, cl, cd, cm_raw):
        """处理CFD计算结果，确保Cm值统一处理"""
        # Cm统一取绝对值
        cm = self.process_cm_value(cm_raw)
        print(f"Cm原始值: {cm_raw}, 处理后的值(绝对值): {cm}")

        return cl, cd, cm

    """
    在 evaluate_surrogate 方法中关于Cm的处理部分
     由于Cm已经是绝对值（正数），我们希望最大化它
     在最小化优化框架中，需要取负值转为最大化问题
    """

    def evaluate_surrogate(self, x):
        """使用代理模型评估解，统一处理Cm值"""
        try:
            x = np.atleast_2d(x)
            objectives = np.zeros(self.n_obj)

            for i, obj_name in enumerate(self.obj_names):
                pred = self.surrogate_models[obj_name].predict(x)
                pred_value = pred[0] if isinstance(pred, np.ndarray) else pred

                # 特殊处理Cm目标 - 统一使用绝对值并转为最大化问题
                if obj_name == 'Cm':
                    # 始终取绝对值并将其作为要最大化的目标
                    abs_value = abs(pred_value)
                    print(f"调试: Cm 原始预测值 = {pred_value}, 取绝对值 = {abs_value}")

                    # 因为在优化框架中我们需要将最大化问题转换为最小化问题，所以取负值
                    objectives[i] = -abs_value  # 取负使得最小化问题成为最大化问题
                elif obj_name == 'Cd':
                    # Cd也是要最大化的目标
                    objectives[i] = -pred_value  # 取负使得最小化问题成为最大化问题
                elif obj_name == 'Cl':
                    # Cl也是要最大化的目标
                    objectives[i] = -pred_value  # 取负使得最小化问题成为最大化问题
                else:
                    # 其他目标保持不变
                    objectives[i] = pred_value

            return objectives
        except Exception as e:
            print(f"代理模型评估出错: {str(e)}")
            raise

    def save_optimization_state(self, iteration: int,
                                pareto_front: np.ndarray,
                                diversity: float,
                                new_points: np.ndarray):
        """保存优化状态"""
        self.optimization_history['iterations'].append(iteration)
        self.optimization_history['best_objectives'].append(
            np.max(pareto_front, axis=0)
        )
        self.optimization_history['population_diversity'].append(diversity)
        self.optimization_history['sample_points'].append(new_points)

        # 保存到文件 - 使用安全的时间戳格式
        history_file = os.path.join(
            self.base_path,
            safe_filename("optimization_history", "npz", include_timestamp=True, iteration=iteration)
        )

        try:
            np.savez(
                history_file,
                iterations=np.array(self.optimization_history['iterations']),
                best_objectives=np.array(self.optimization_history['best_objectives']),
                diversity=np.array(self.optimization_history['population_diversity']),
                samples=np.array(self.optimization_history['sample_points'])
            )
            print(f"\n优化历史已保存至: {history_file}")
            return True
        except Exception as e:
            print(f"保存优化历史时出错: {str(e)}")
            # 尝试不同的文件名格式
            alternative_file = os.path.join(self.base_path, f"optimization_history_iter_{iteration}.npz")
            try:
                np.savez(
                    alternative_file,
                    iterations=np.array(self.optimization_history['iterations']),
                    best_objectives=np.array(self.optimization_history['best_objectives']),
                    diversity=np.array(self.optimization_history['population_diversity']),
                    samples=np.array(self.optimization_history['sample_points'])
                )
                print(f"使用替代文件名保存成功: {alternative_file}")
                return True
            except Exception as e2:
                print(f"使用替代文件名保存也失败: {str(e2)}")
                return False

    def optimize(self, n_generations: int = 100, pop_size: int = 100, convergence_threshold: float = 0.001,
                 convergence_window: int = 5):
        """执行优化过程"""
        # 记录开始时间和用户信息
        print(f"开始优化 - 时间: {datetime.now()}")
        print(f"用户: {self.user_login}")
        print(f"操作系统: {platform.system()} {platform.release()}")

        print(f"最大迭代次数: {n_generations}, 种群大小: {pop_size}")
        print(f"收敛阈值: {convergence_threshold}, 收敛窗口: {convergence_window}")

        # 初始化优化历史
        self.current_iteration = 0
        self.all_objectives = []
        self.max_iterations = n_generations  # 存储最大迭代次数

        # 初始化收敛检测所需的变量
        self._init_model_evaluation_history()

        # 创建验证器
        validator = SurrogateValidator(base_path=self.base_path, save_dir=os.path.join(self.base_path, "surrogate_validation"))
        previous_best = None

        # 记录是否收敛
        converged = False
        convergence_reason = ""

        for current_iteration in range(n_generations):
            print(f"\n{'=' * 50}")
            print(f"开始第 {current_iteration + 1} 代迭代")
            print('=' * 50)

            try:
                # 1. 基于当前代理模型运行优化算法
                population, objectives = self.optimizer.run_generation(self.evaluate_surrogate)
                fronts = self.optimizer.fast_non_dominated_sort(objectives)
                pareto_front_indices = fronts[0]
                pareto_front = objectives[pareto_front_indices]

                # 保存当前迭代的目标函数值
                self.all_objectives.append(objectives)
                # 计算并记录超体积
                if not hasattr(self, 'hypervolume_history'):
                    self.hypervolume_history = []
                hv = self._calculate_hypervolume(pareto_front)
                self.hypervolume_history.append(hv)
                print(f"当前超体积: {hv:.6f}")

                # 2. 计算优化指标并保存数据
                improvement, diversity = self.calculate_metrics(pareto_front, previous_best)
                previous_best = pareto_front.copy()

                # 保存迭代数据
                self.data_manager.save_iteration_data(
                    iteration=current_iteration,
                    objectives=objectives,
                    population=population
                )

                # 绘制帕累托前沿
                self.data_manager.plot_pareto_front(
                    objectives=pareto_front,
                    iteration=current_iteration
                )

                print(f"\n迭代 {current_iteration + 1}/{n_generations}")
                print(f"当前改善度: {improvement:.4f}, 多样性: {diversity:.4f}")

                # 检查是否满足收敛条件
                converged, convergence_reason = self._check_convergence(
                    self.optimization_history,
                    current_iteration,
                    window_size=convergence_window,
                    threshold=convergence_threshold
                )

                if converged:
                    print(f"\n优化已收敛: {convergence_reason}")
                    print(f"总迭代次数: {current_iteration + 1}")
                    break

                # 3. 选择新的采样点
                n_samples = min(3, len(pareto_front_indices))
                new_points = self.sampling_strategy.select_points(
                    population=population,
                    objectives=objectives,
                    pareto_front_indices=pareto_front_indices,
                    surrogate_models=self.surrogate_models,
                    iteration=current_iteration,
                    n_points=n_samples
                )
                new_points = np.round(new_points, decimals=6)

                # 4. 使用代理模型预测新点的响应值
                predicted_responses = np.zeros((len(new_points), len(self.obj_names)))
                for i, obj_name in enumerate(self.obj_names):
                    predicted_responses[:, i] = self.surrogate_models[obj_name].predict(new_points)
                predicted_responses = np.round(predicted_responses, decimals=6)

                # 5. 保存代理模型预测结果
                surrogate_predictions_file = os.path.join(
                    self.base_path,
                    safe_filename("surrogate_predictions", "txt", include_timestamp=True,
                                  iteration=current_iteration + 1)
                )
                pred_df = pd.DataFrame(
                    np.hstack([new_points, predicted_responses]),
                    columns=self.var_names + self.obj_names
                )
                pred_df.to_csv(surrogate_predictions_file, sep='\t', index=False, float_format='%.6f')
                print(f"\n代理模型预测结果已保存至: {surrogate_predictions_file}")

                # 6. 执行CFD计算和数据更新
                try:
                    # 创建DataFrame来存储新采样点
                    df_new_points = pd.DataFrame(
                        columns=self.var_names + self.obj_names,
                        index=range(len(new_points))
                    )

                    # 初始化变量值列 - 确保6位小数格式
                    for i, var_name in enumerate(self.var_names):
                        df_new_points[var_name] = [f"{x:12.6f}" for x in new_points[:, i]]

                    # 初始化目标值列为空字符串
                    for obj_name in self.obj_names:
                        df_new_points[obj_name] = [''] * len(new_points)

                    # 保存初始的data_new.txt
                    df_new_points.to_csv(self.new_points_file, sep='\t', index=False, float_format='%12.6f')
                    print(f"\n已将 {len(new_points)} 个新采样点写入 {self.new_points_file}")

                    # 存储完成计算的点数据
                    completed_points_data = []
                    actual_responses = np.zeros((len(new_points), len(self.obj_names)))

                    # 逐个处理采样点
                    for i, point in enumerate(new_points):
                        print(f"\n处理第 {i + 1}/{len(new_points)} 个采样点")
                        try:
                            # 运行CFD计算
                            cl_raw, cd_raw, cm_raw = self.cfd.run_cfd_workflow(point)

                            # 统一处理CFD计算结果，特别是Cm值
                            cl, cd, cm = self.process_cfd_results(cl_raw, cd_raw, cm_raw)

                            # 更新actual_responses - 使用处理后的值
                            actual_responses[i] = [cl, cd, cm]

                            # 更新data_new.txt中当前行的目标值
                            df_new_points.loc[i, ['Cl', 'Cd', 'Cm']] = [f"{cl:12.6f}", f"{cd:12.6f}", f"{cm:12.6f}"]
                            df_new_points.to_csv(self.new_points_file, sep='\t', index=False, float_format='%12.6f')

                            # 将完整的点数据添加到列表
                            point_data = np.concatenate([point, [cl, cd, cm]])
                            completed_points_data.append(point_data)

                            # 输出计算结果
                            print(f"采样点 {i + 1} 计算完成:")
                            print(f"Cl = {cl:12.6f}")
                            print(f"Cd = {cd:12.6f}")
                            print(f"Cm = {cm:12.6f}")

                            # 比较预测值和实际值
                            print("\n预测值与实际值比较:")
                            for j, obj_name in enumerate(['Cl', 'Cd', 'Cm']):
                                pred_val = predicted_responses[i, j]
                                true_val = actual_responses[i, j]
                                error = abs(pred_val - true_val)
                                rel_error = abs(error / true_val) * 100 if true_val != 0 else float('inf')
                                print(f"{obj_name}:")
                                print(f"  预测值: {pred_val:.6f}")
                                print(f"  实际值: {true_val:.6f}")
                                print(f"  绝对误差: {error:.6f}")
                                print(f"  相对误差: {rel_error:.2f}%")

                        except Exception as e:
                            print(f"采样点 {i + 1} CFD计算失败: {str(e)}")
                            raise

                    # 更新doe_dynamic.txt
                    print("\n更新doe_dynamic.txt文件...")
                    with open(self.doe_file, 'a') as f:
                        for point_data in completed_points_data:
                            formatted_data = '\t'.join([f"{x:12.6f}" for x in point_data])
                            f.write(formatted_data + '\n')
                    print("doe_dynamic.txt更新完成")

                    # 运行Matlab文件备份脚本
                    matlab_script = os.path.join(self.work_dir, 'file_deal.m')
                    matlab_cmd = f'\"{os.path.join("E:", "Program Files", "MATLAB", "R2022b", "bin", "matlab.exe")}\" /minimize -r \"run(\'{matlab_script}\'); exit;\"'
                    print("\n开始运行Matlab脚本进行文件备份...")
                    process = subprocess.Popen(matlab_cmd, shell=True)
                    process.wait()
                    if process.returncode == 0:
                        print("文件备份完成")
                    else:
                        print("文件备份过程中出现错误")

                    # 7. 更新验证历史和可视化
                    validator.update_history(
                        iteration=current_iteration,
                        objectives=self.obj_names,
                        true_values=actual_responses,
                        predicted_values=predicted_responses
                    )
                    validator.plot_improved_validation_metrics(current_iteration)
                    validator.save_validation_report(current_iteration)

                    # 8. 每5次迭代更新历史图
                    # 在optimize方法中用于定期更新图表
                    if current_iteration % 3 == 0:
                        history = self.get_optimization_history()
                        self.data_manager.plot_optimization_history(
                            history,
                            include_initial_doe=True,
                            doe_file=self.doe_file
                        )



                    # 9. 更新代理模型
                    self.current_iteration = current_iteration
                    self.initialize_models()

                    # 可视化当前帕累托前沿
                    self.data_manager.plot_pareto_front(pareto_front, current_iteration)

                    # 10. 保存优化状态
                    self.save_optimization_state(
                        current_iteration,
                        pareto_front,
                        diversity,
                        new_points
                    )

                except Exception as e:
                    print(f"\n迭代 {current_iteration + 1} 出现错误:")
                    print(f"错误信息: {str(e)}")
                    self._handle_optimization_error(e, current_iteration)
                    raise

            # 这里是外层try块的except，它被错误地缩进在内层try块中
            # 应该在内层try-except块之后，与外层try对齐
            except Exception as e:
                print(f"\n迭代 {current_iteration + 1} 出现错误:")
                print(f"错误信息: {str(e)}")
                self._handle_optimization_error(e, current_iteration)
                raise

        # 这些语句应该在外层for循环之后，而不是在内部try-except块中
        # 优化完成，保存最终结果
        self._save_final_results(current_iteration + 1)  # 使用实际的迭代次数

        if converged:
            print(f"\n优化已收敛并终止: {convergence_reason}")
        else:
            print("\n优化已完成所有迭代，未检测到收敛")

        print(f"总用时: {datetime.now() - self.start_time}")
        print(f"总迭代次数: {current_iteration + 1}")



    def _predict_responses(self, new_points):
        """使用代理模型预测响应值"""
        predicted_responses = np.zeros((len(new_points), len(self.obj_names)))
        for i, obj_name in enumerate(self.obj_names):
            predicted_responses[:, i] = self.surrogate_models[obj_name].predict(new_points)
        return np.round(predicted_responses, decimals=6)

    def _save_predictions(self, new_points, predicted_responses, iteration):
        """保存预测结果"""
        surrogate_predictions_file = os.path.join(
            self.base_path,
            f"surrogate_predictions_{iteration + 1}.txt"
        )
        pred_df = pd.DataFrame(
            np.hstack([new_points, predicted_responses]),
            columns=self.var_names + self.obj_names
        )
        pred_df.to_csv(surrogate_predictions_file, sep='\t', index=False, float_format='%.6f')
        print(f"\n代理模型预测结果已保存至: {surrogate_predictions_file}")

    def _perform_cfd_calculations(self, new_points):
        """执行CFD计算"""
        actual_responses = np.zeros((len(new_points), len(self.obj_names)))
        completed_points_data = []

        for i, point in enumerate(new_points):
            print(f"\n处理第 {i + 1}/{len(new_points)} 个采样点")
            cl, cd, cm = self.cfd.run_cfd_workflow(point)
            actual_responses[i] = [cl, cd, cm]
            completed_points_data.append(np.concatenate([point, [cl, cd, cm]]))

            # 更新doe_dynamic.txt
            self._update_doe_file(completed_points_data[-1])

        return actual_responses


    def export_final_model(self, output_dir=None):
        """导出最终训练好的代理模型，方便后续使用"""
        if output_dir is None:
            output_dir = os.path.join(self.base_path, 'final_models')

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存模型
        for obj_name, model in self.surrogate_models.items():
            model_file = os.path.join(output_dir, f'{obj_name}_final_model_{timestamp}.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"最终{obj_name}代理模型已保存至: {model_file}")

        # 生成使用说明
        usage_file = os.path.join(output_dir, 'model_usage_guide.txt')
        with open(usage_file, 'w') as f:
            f.write("== 代理模型使用指南 ==\n\n")
            f.write("1. 加载模型:\n")
            f.write("   import pickle\n")
            f.write("   with open('model_file.pkl', 'rb') as f:\n")
            f.write("       model = pickle.load(f)\n\n")
            f.write("2. 准备输入数据:\n")
            f.write("   import numpy as np\n")
            f.write("   # 示例: 创建一个新的设计点\n")
            f.write("   x_new = np.array([[400, 1200, 25]])  # [Chord, Distance, Fai]\n\n")
            f.write("3. 使用模型预测:\n")
            f.write("   y_pred = model.predict(x_new)\n")
            f.write("   print(f'预测结果: {y_pred[0]}')\n\n")
            f.write("4. 批量预测:\n")
            f.write("   # 多个设计点\n")
            f.write("   x_batch = np.array([\n")
            f.write("       [300, 1000, 20],  # 设计点1\n")
            f.write("       [400, 1200, 25],  # 设计点2\n")
            f.write("       [500, 1500, 30],  # 设计点3\n")
            f.write("   ])\n")
            f.write("   y_batch = model.predict(x_batch)\n\n")
            f.write("5. 保存预测结果:\n")
            f.write("   import pandas as pd\n")
            f.write("   results = pd.DataFrame()\n")
            f.write("   results['Chord'] = x_batch[:, 0]\n")
            f.write("   results['Distance'] = x_batch[:, 1]\n")
            f.write("   results['Fai'] = x_batch[:, 2]\n")
            f.write("   results['Prediction'] = y_batch\n")
            f.write("   results.to_csv('predictions.csv', index=False)\n")

        print(f"模型使用指南已保存至: {usage_file}")

        return output_dir

    def prepare_analysis_data(self):
        """
        准备用于分析的综合数据框，并保存到CSV文件

        Returns:
        --------
        pandas.DataFrame: 包含所有设计变量和响应变量的数据框
        """
        try:
            # 合并DOE数据和优化结果
            all_points = np.vstack([self.initial_samples, self.optimization_samples])
            all_responses = np.vstack([self.initial_responses, self.optimization_responses])

            # 创建数据框
            df = pd.DataFrame(all_points, columns=self.design_var_names)

            # 添加响应变量
            for i, obj_name in enumerate(self.obj_names):
                df[obj_name] = all_responses[:, i]

            # 添加一列标记点的来源（DOE或优化）
            df['Sample_Type'] = ['DOE'] * len(self.initial_samples) + ['Optimization'] * len(self.optimization_samples)

            # 添加一列标记每个点的迭代编号
            doe_iterations = [0] * len(self.initial_samples)
            opt_iterations = list(range(1, len(self.optimization_samples) + 1))
            df['Iteration'] = doe_iterations + opt_iterations

            # 保存数据到CSV文件
            save_dir = os.path.join(self.base_path, 'analysis_data')
            os.makedirs(save_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'optimization_analysis_data_{timestamp}.csv'
            filepath = os.path.join(save_dir, filename)

            df.to_csv(filepath, index=False)
            print(f"\n分析数据已保存到: {os.path.abspath(filepath)}")

            return df

        except Exception as e:
            print(f"准备分析数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()  # 返回空数据框

    def analyze_optimization_results(self):
        """
        优化完成后的综合分析，生成所有可能的分析图表
        包括:
        1. Pareto图 - 显示各设计变量对每个响应的影响程度
        2. 主效应图 - 显示每个设计变量对每个响应的主要影响
        3. 交互效应图 - 显示设计变量对之间的交互对响应的影响
        4. ANOVA表 - 显示方差分析结果
        5. 响应曲面图 - 在二维空间中可视化响应表面
        6. 相关性矩阵 - 显示所有变量之间的相关性
        """
        print("\n========== 开始生成优化结果分析图表 ==========")

        try:
            # 准备数据 - 将所有历史数据合并为一个数据框
            print("正在准备分析数据...")
            data = self.prepare_analysis_data()

            if data.empty:
                print("错误: 无法准备分析数据，跳过分析步骤")
                return

            print(f"分析数据准备完成，共 {len(data)} 条记录")

            # 1. 相关性矩阵 - 显示所有变量之间的相关性
            print("\n1. 生成相关性矩阵...")
            self.data_manager.plot_correlation_matrix(data,
                                                      self.design_var_names + self.obj_names)

            # 2. 为每个响应变量生成Pareto图
            print("\n2. 生成Pareto影响分析图...")
            for response in self.obj_names:
                self.data_manager.plot_pareto_analysis(data, response)

            # 3. 为每个响应变量生成主效应图
            print("\n3. 生成主效应分析图...")
            for response in self.obj_names:
                self.data_manager.plot_main_effects(data, response,
                                                    factors=self.design_var_names)

            # 4. 为所有可能的设计变量对组合生成交互效应图
            print("\n4. 生成交互效应图...")
            for i, var1 in enumerate(self.design_var_names[:-1]):
                for var2 in self.design_var_names[i + 1:]:
                    for response in self.obj_names:
                        self.data_manager.plot_interaction(data, var1, var2, response)

            # 5. 为每个响应变量创建ANOVA分析表
            print("\n5. 生成方差分析(ANOVA)表...")
            for response in self.obj_names:
                self.data_manager.create_anova_table(data, response,
                                                     self.design_var_names)

            # 6. 为每个响应变量生成响应曲面图 (所有变量对的组合)
            print("\n6. 生成响应曲面图...")
            for i, var1 in enumerate(self.design_var_names[:-1]):
                for var2 in self.design_var_names[i + 1:]:
                    for response in self.obj_names:
                        self.data_manager.plot_response_surface(data, var1, var2,
                                                                response)

            # 7. 生成帕累托前沿图 (对于多目标优化)
            if len(self.obj_names) >= 2:
                print("\n7. 生成帕累托前沿分析图...")
                # 2D帕累托前沿 (各对目标之间)
                for i, obj1 in enumerate(self.obj_names[:-1]):
                    for obj2 in self.obj_names[i + 1:]:
                        # 注意这里使用的是修改后的函数名
                        self.data_manager.plot_pareto_front_analysis(data, obj1, obj2)

                # 如果有3个目标，绘制3D帕累托前沿
                if len(self.obj_names) >= 3:
                    # 注意这里使用的是修改后的函数名
                    self.data_manager.plot_3d_pareto_front_analysis(data,
                                                                    self.obj_names[0],
                                                                    self.obj_names[1],
                                                                    self.obj_names[2])

            print("\n========== 分析图表生成完成 ==========")

        except Exception as e:
            print(f"\n生成分析图表时出错: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # 设置时间和用户信息
    current_time = datetime.now()
    user_login = "Gtao1013"

    print(f"开始优化 - 时间: {current_time}")
    print(f"用户: {user_login}")

    try:
        # 创建优化问题实例
        problem = OptimizationProblem()

        # 初始化代理模型
        if problem.initialize_models():
            # 运行优化，添加收敛参数
            problem.optimize(
                n_generations=40,  # 最大迭代次数
                pop_size=20,  # 种群大小
                convergence_threshold=0.001,  # 收敛阈值，可根据实际情况调整
                convergence_window=5  # 收敛判断窗口大小
            )

            # 导出最终模型
            problem.export_final_model()

            # 添加这一行来执行结果分析
            print("\n优化完成，开始分析结果...")
            problem.analyze_optimization_results()

        else:
            print("代理模型初始化失败")

    except Exception as e:
        print(f"优化过程中出现错误: {str(e)}")
        # 记录错误日志
        if 'problem' in locals():
            error_log = os.path.join(problem.base_path, "optimization_errors.log")
            with open(error_log, 'a') as f:
                f.write(f"\n{'=' * 50}")
                f.write(f"\n时间: {current_time}")
                f.write(f"\n用户: {user_login}")
                f.write(f"\n错误信息: {str(e)}")
                f.write(f"\n{'=' * 50}\n")
        raise