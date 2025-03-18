# data_manager.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import pickle
import glob
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, max_error, mean_absolute_error
from utils import safe_filename
from sklearn.gaussian_process import GaussianProcessRegressor
import statsmodels.api as sm
from statsmodels.formula.api import ols
import traceback
from matplotlib.ticker import MaxNLocator
from surrogate_model import KrigingModel
import seaborn as sns
from scipy.interpolate import griddata

class OptimizationDataManager:
    def __init__(self, base_path):
        self.base_path = base_path
        self.obj_names = ['Cl', 'Cd', 'Cm']  # 设置目标名称

        # 创建必要的目录
        self.create_directories()

    def create_directories(self):
        """创建必要的目录结构"""
        directories = [
            os.path.join(self.base_path, 'optimization_plots'),
            os.path.join(self.base_path, 'model_validation'),
            os.path.join(self.base_path, 'surrogate_models'),
            os.path.join(self.base_path, 'surrogate_validation'),
            os.path.join(self.base_path, 'iteration_data')
        ]

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"创建目录: {directory}")

    def save_validation_plot(self, y_true, y_pred, fold, obj_name, iteration):
        """保存验证结果图"""
        try:
            # 创建保存目录
            save_dir = os.path.join(self.base_path, 'model_validation')
            os.makedirs(save_dir, exist_ok=True)

            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.7, color='blue')

            # 添加对角线
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            # 添加标题和标签
            plt.title(f'{obj_name} 代理模型验证 - 折 {fold}', fontsize=14)
            plt.xlabel('真实值', fontsize=12)
            plt.ylabel('预测值', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)

            # 添加R²值到图中
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
                     transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.8),
                     fontsize=12)

            # 保存图片
            filename = f'{obj_name}_iter_{iteration}_fold_{fold}_validation.png'
            filepath = os.path.join(save_dir, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300)
            plt.close()

            print(f"验证图已保存至: {filepath}")

        except Exception as e:
            print(f"保存验证图时出错: {str(e)}")

    def save_surrogate_model(self, model, obj_name, iteration):
        """保存代理模型及其参数"""
        try:
            # 创建保存目录
            save_dir = os.path.join(self.base_path, 'surrogate_models')
            os.makedirs(save_dir, exist_ok=True)

            # 保存模型
            model_file = os.path.join(save_dir, safe_filename(f'{obj_name}_model', 'pkl', include_timestamp=True,
                                                              iteration=iteration))
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)

            # 提取并保存模型参数
            model_params = {}

            # 通用信息
            model_params['模型类型'] = type(model).__name__
            model_params['迭代次数'] = iteration
            model_params['目标名称'] = obj_name

            # 尝试提取GPR相关参数
            gpr = None
            if hasattr(model, 'gpr'):
                gpr = model.gpr
            elif hasattr(model, 'model'):
                gpr = model.model
            elif isinstance(model, GaussianProcessRegressor):
                gpr = model

            # 如果找到GPR模型，提取其核函数和参数
            if gpr is not None:
                if hasattr(gpr, 'kernel_'):
                    model_params['核函数表达式'] = str(gpr.kernel_)

                    # 提取核函数的详细参数
                    if hasattr(gpr.kernel_, 'get_params'):
                        kernel_params = gpr.kernel_.get_params()
                        for param_name, param_value in kernel_params.items():
                            if isinstance(param_value, (int, float, bool, str, list, tuple)):
                                model_params[f'核参数_{param_name}'] = param_value

                    # 如果是复合核，尝试提取各个子核的参数
                    if hasattr(gpr.kernel_, 'k1'):
                        model_params['k1_表达式'] = str(gpr.kernel_.k1)
                        if hasattr(gpr.kernel_.k1, 'constant_value'):
                            model_params['信号方差'] = gpr.kernel_.k1.constant_value

                    if hasattr(gpr.kernel_, 'k2'):
                        model_params['k2_表达式'] = str(gpr.kernel_.k2)
                        if hasattr(gpr.kernel_.k2, 'length_scale'):
                            model_params['长度尺度'] = gpr.kernel_.k2.length_scale.tolist() if hasattr(
                                gpr.kernel_.k2.length_scale, 'tolist') else gpr.kernel_.k2.length_scale
                        if hasattr(gpr.kernel_.k2, 'nu'):
                            model_params['nu参数'] = gpr.kernel_.k2.nu

                # 提取其他GPR参数
                if hasattr(gpr, 'alpha'):
                    model_params['alpha'] = gpr.alpha
                if hasattr(gpr, 'X_train_'):
                    model_params['训练样本数'] = len(gpr.X_train_)
                if hasattr(gpr, 'y_train_'):
                    model_params['y_train_范围'] = [min(gpr.y_train_), max(gpr.y_train_)]
                    model_params['y_train_均值'] = float(np.mean(gpr.y_train_))
                    model_params['y_train_方差'] = float(np.var(gpr.y_train_))

            # 保存参数到文本文件
            params_file = os.path.join(save_dir, safe_filename(f'{obj_name}_params', 'txt', include_timestamp=True,
                                                               iteration=iteration))
            with open(params_file, 'w', encoding='utf-8') as f:
                f.write(f"代理模型参数 - {obj_name} - 迭代 {iteration}\n")
                f.write(f"时间: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n\n")

                # 写入GPR模型的数学表达式
                if '核函数表达式' in model_params:
                    f.write(f"\n数学表达式:\n")
                    f.write(f"{model_params['核函数表达式']}\n\n")

                # 写入所有参数
                f.write("详细参数:\n")
                for param_name, param_value in model_params.items():
                    if param_name != '核函数表达式':  # 避免重复
                        f.write(f"{param_name}: {param_value}\n")

                # 添加GPR预测函数的数学表达式
                f.write("\n\n高斯过程回归预测函数:\n")
                f.write("μ(x*) = k(x*, X)[K(X,X)+σn²I]⁻¹y\n")
                f.write("σ²(x*) = k(x*,x*) - k(x*,X)[K(X,X)+σn²I]⁻¹k(X,x*)\n")

                # 添加Matern核函数的数学表达式
                if any('Matern' in str(param_value) for param_value in model_params.values() if
                       isinstance(param_value, str)):
                    nu_value = model_params.get('nu参数', 1.5)
                    f.write(f"\nMatern核函数(nu={nu_value})的数学表达式:\n")
                    if nu_value == 1.5:
                        f.write("k(x_i,x_j) = (1 + √3|x_i-x_j|/l) * exp(-√3|x_i-x_j|/l)\n")
                    elif nu_value == 2.5:
                        f.write("k(x_i,x_j) = (1 + √5|x_i-x_j|/l + 5|x_i-x_j|²/(3l²)) * exp(-√5|x_i-x_j|/l)\n")
                    else:
                        f.write("k(x_i,x_j) = 2^(1-ν)/Γ(ν) * (√2ν|x_i-x_j|/l)^ν * K_ν(√2ν|x_i-x_j|/l)\n")
                        f.write("其中K_ν是修正贝塞尔函数，Γ是gamma函数\n")

            print(f"{obj_name}代理模型已保存至: {model_file}")
            print(f"{obj_name}代理模型参数已保存至: {params_file}")

        except Exception as e:
            print(f"保存代理模型时出错: {str(e)}")
            traceback.print_exc()

    def save_iteration_data(self, iteration, objectives, population):
        """保存迭代数据"""
        try:
            # 创建保存目录
            save_dir = os.path.join(self.base_path, 'iteration_data')
            os.makedirs(save_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存目标函数值
            objectives_file = os.path.join(save_dir, safe_filename('objectives', 'csv', include_timestamp=True, iteration=iteration))
            obj_df = pd.DataFrame(objectives, columns=self.obj_names)
            obj_df.to_csv(objectives_file, index=False)

            # 保存种群
            population_file = os.path.join(save_dir, safe_filename('population', 'csv', include_timestamp=True, iteration=iteration))
            pop_df = pd.DataFrame(population, columns=['Chord', 'Distance', 'Fai'])
            pop_df.to_csv(population_file, index=False)

            print(f"迭代{iteration}的目标函数值已保存至: {objectives_file}")
            print(f"迭代{iteration}的种群已保存至: {population_file}")

        except Exception as e:
            print(f"保存迭代数据时出错: {str(e)}")

    def plot_pareto_analysis(self, data, response_name, num_factors=None):
        """
        为指定的响应变量创建Pareto图，展示各因素的影响大小

        Parameters:
        -----------
        data : pandas.DataFrame
            包含因素(自变量)和响应(因变量)的数据表
        response_name : str
            响应变量的名称
        num_factors : int, optional
            要显示的主要因素数量，默认显示全部
        """
        try:
            # 确保目录存在
            save_dir = os.path.join(self.base_path, 'analysis_plots')
            os.makedirs(save_dir, exist_ok=True)

            # 获取自变量列名(排除响应变量)
            factor_cols = [col for col in data.columns if col != response_name]

            # 计算每个因素对响应的相关性(绝对值)
            correlations = []
            for factor in factor_cols:
                corr = np.abs(np.corrcoef(data[factor], data[response_name])[0, 1])
                correlations.append((factor, corr))

            # 按相关性大小排序
            correlations.sort(key=lambda x: x[1], reverse=True)

            # 限制显示的因素数量
            if num_factors is not None and num_factors < len(correlations):
                correlations = correlations[:num_factors]

            # 创建Pareto图
            plt.figure(figsize=(10, 6))

            factors = [item[0] for item in correlations]
            values = [item[1] for item in correlations]

            # 绘制条形图
            bars = plt.bar(factors, values, color='steelblue')

            # 添加累积线
            cumulative = np.cumsum(values) / np.sum(values) * 100
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.plot(factors, cumulative, 'ro-', linewidth=2)
            ax2.set_ylim([0, 105])

            # 设置标签和标题
            ax1.set_xlabel('因素', fontsize=12)
            ax1.set_ylabel('相关系数(绝对值)', fontsize=12)
            ax2.set_ylabel('累积百分比(%)', fontsize=12)
            plt.title(f'{response_name}的Pareto分析', fontsize=14)

            # 旋转x轴标签以防止重叠
            plt.xticks(rotation=45, ha='right')

            # 为条形添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom')

            plt.tight_layout()

            # 保存图表
            filename = safe_filename(f'pareto_analysis_{response_name}', 'png', include_timestamp=True)
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()
            print(f"{response_name}的Pareto分析图已保存至: {filepath}")

            return filepath

        except Exception as e:
            print(f"创建Pareto分析图时出错: {str(e)}")
            traceback.print_exc()
            return None

    def plot_main_effects(self, data, response_name, factors=None):
        """
        为指定的响应变量创建主效应图，展示各因素对响应的影响

        Parameters:
        -----------
        data : pandas.DataFrame
            包含因素(自变量)和响应(因变量)的数据表
        response_name : str
            响应变量的名称
        factors : list, optional
            要分析的因素列表，默认分析所有非响应变量
        """
        try:
            # 确保目录存在
            save_dir = os.path.join(self.base_path, 'analysis_plots')
            os.makedirs(save_dir, exist_ok=True)

            # 如果未指定因素，使用所有非响应变量列
            if factors is None:
                factors = [col for col in data.columns if col != response_name]

            # 计算需要的子图行数和列数
            n_factors = len(factors)
            n_cols = min(3, n_factors)  # 最多3列
            n_rows = (n_factors + n_cols - 1) // n_cols  # 向上取整计算行数

            plt.figure(figsize=(5 * n_cols, 4 * n_rows))

            # 为每个因素创建主效应图
            for i, factor in enumerate(factors, 1):
                plt.subplot(n_rows, n_cols, i)

                # 计算分组的均值
                # 对连续变量，创建10个等宽分组
                if data[factor].nunique() > 10:
                    # 创建等宽分组
                    bins = np.linspace(data[factor].min(), data[factor].max(), 11)
                    labels = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
                    data['_bin'] = pd.cut(data[factor], bins=bins, labels=labels)

                    # 计算每个分组的平均响应值
                    means = data.groupby('_bin')[response_name].mean()
                    plt.plot(means.index, means.values, 'bo-', linewidth=2)

                    # 删除临时列
                    data.drop('_bin', axis=1, inplace=True)
                else:
                    # 对于离散变量，直接按值分组
                    means = data.groupby(factor)[response_name].mean()
                    plt.bar(means.index.astype(str), means.values, color='steelblue')

                # 添加水平参考线（总体平均值）
                plt.axhline(y=data[response_name].mean(), color='red', linestyle='--')

                # 设置标签
                plt.xlabel(factor, fontsize=10)
                plt.ylabel(response_name, fontsize=10)
                plt.title(f'{factor}的主效应', fontsize=12)

                # 如果是离散分类变量且类别较多，旋转标签
                if data[factor].nunique() > 5 and data[factor].nunique() <= 10:
                    plt.xticks(rotation=45, ha='right')

            # 调整子图布局
            plt.tight_layout()

            # 保存图表
            filename = safe_filename(f'main_effects_{response_name}', 'png', include_timestamp=True)
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()
            print(f"{response_name}的主效应图已保存至: {filepath}")

            return filepath

        except Exception as e:
            print(f"创建主效应图时出错: {str(e)}")
            traceback.print_exc()
            return None

    def plot_interaction(self, data, factor1, factor2, response_name):
        """
        创建两个因素交互作用图

        Parameters:
        -----------
        data : pandas.DataFrame
            包含因素和响应的数据表
        factor1 : str
            第一个因素(自变量)的名称
        factor2 : str
            第二个因素(自变量)的名称
        response_name : str
            响应变量(因变量)的名称
        """
        try:
            # 确保目录存在
            save_dir = os.path.join(self.base_path, 'analysis_plots')
            os.makedirs(save_dir, exist_ok=True)

            plt.figure(figsize=(10, 8))

            # 处理连续变量
            if data[factor1].nunique() > 10:
                # 创建分组
                bins1 = pd.qcut(data[factor1], q=5, duplicates='drop')
                # 获取每个分组的中间值
                unique_bins = sorted(bins1.unique())
                labels1 = [f"{b.left:.2f}-{b.right:.2f}" for b in unique_bins]
            else:
                # 对于离散变量直接使用值
                bins1 = data[factor1]
                labels1 = sorted(data[factor1].unique())

            if data[factor2].nunique() > 10:
                # 创建分组
                bins2 = pd.qcut(data[factor2], q=5, duplicates='drop')
                # 获取每个分组的中间值
                unique_bins = sorted(bins2.unique())
                labels2 = [f"{b.left:.2f}-{b.right:.2f}" for b in unique_bins]
            else:
                # 对于离散变量直接使用值
                bins2 = data[factor2]
                labels2 = sorted(data[factor2].unique())

            # 使用分组变量计算平均响应
            df = pd.DataFrame({
                'factor1': bins1,
                'factor2': bins2,
                'response': data[response_name]
            })

            # 计算每个组合的平均值
            interaction_data = df.groupby(['factor1', 'factor2'])['response'].mean().reset_index()

            # 将结果转换为交互图
            pivot_table = interaction_data.pivot_table(index='factor1', columns='factor2', values='response')

            # 绘制交互图
            colors = plt.cm.viridis(np.linspace(0, 1, len(pivot_table.columns)))

            for i, col in enumerate(pivot_table.columns):
                plt.plot(pivot_table.index.astype(str), pivot_table[col],
                         marker='o', color=colors[i], label=f'{factor2}={col}')

            plt.xlabel(factor1, fontsize=12)
            plt.ylabel(response_name, fontsize=12)
            plt.title(f'{factor1}和{factor2}的交互效应对{response_name}的影响', fontsize=14)
            plt.legend(title=factor2, loc='best')

            # 如果有多个类别，可能需要旋转x轴标签
            if len(pivot_table.index) > 5:
                plt.xticks(rotation=45, ha='right')

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            # 保存图表
            filename = safe_filename(f'interaction_{factor1}_{factor2}_{response_name}', 'png', include_timestamp=True)
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()
            print(f"{factor1}和{factor2}的交互效应图已保存至: {filepath}")

            return filepath

        except Exception as e:
            print(f"创建交互效应图时出错: {str(e)}")
            traceback.print_exc()
            return None

    def create_anova_table(self, data, response_name, factors):
        """
        创建响应变量的方差分析(ANOVA)表，并将结果保存为图表和CSV

        Parameters:
        -----------
        data : pandas.DataFrame
            包含因素和响应的数据表
        response_name : str
            响应变量的名称
        factors : list
            要包含在ANOVA分析中的因素列表
        """
        try:
            # 确保目录存在
            save_dir = os.path.join(self.base_path, 'analysis_plots')
            os.makedirs(save_dir, exist_ok=True)

            # 创建用于ANOVA的公式
            formula = f"{response_name} ~ " + " + ".join(factors)

            # 拟合OLS模型
            model = ols(formula, data=data).fit()

            # 执行ANOVA分析
            anova_table = sm.stats.anova_lm(model, typ=2)

            # 添加贡献率列
            ss_total = anova_table['sum_sq'].sum()
            anova_table['contribution'] = anova_table['sum_sq'] / ss_total * 100

            # 保存ANOVA表为CSV
            csv_filename = safe_filename(f'anova_{response_name}', 'csv', include_timestamp=True)
            csv_filepath = os.path.join(save_dir, csv_filename)
            anova_table.to_csv(csv_filepath)
            print(f"ANOVA表已保存至: {csv_filepath}")

            # 创建ANOVA表的可视化
            plt.figure(figsize=(12, len(factors) * 0.7 + 2))

            # 绘制因素贡献率条形图
            factors_to_plot = anova_table.index.tolist()
            if 'Residual' in factors_to_plot:
                factors_to_plot.remove('Residual')

            contributions = anova_table.loc[factors_to_plot, 'contribution']

            # 按贡献率排序
            sorted_indices = contributions.argsort()[::-1]  # 降序排序
            sorted_factors = [factors_to_plot[i] for i in sorted_indices]
            sorted_contribs = [contributions[i] for i in sorted_indices]

            # 绘制条形图
            bars = plt.bar(sorted_factors, sorted_contribs, color='steelblue')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                         f'{height:.2f}%', ha='center', va='bottom', rotation=0)

            plt.xlabel('因素', fontsize=12)
            plt.ylabel('贡献率 (%)', fontsize=12)
            plt.title(f'{response_name}的方差分析(ANOVA)结果', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, max(sorted_contribs) * 1.15)  # 留出空间显示标签
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # 保存图表
            plot_filename = safe_filename(f'anova_plot_{response_name}', 'png', include_timestamp=True)
            plot_filepath = os.path.join(save_dir, plot_filename)
            plt.savefig(plot_filepath, dpi=300)
            plt.close()
            print(f"ANOVA分析图已保存至: {plot_filepath}")

            # 额外绘制一个饼图来展示贡献率
            plt.figure(figsize=(10, 8))

            # 添加残差项
            pie_factors = sorted_factors.copy()
            pie_contribs = sorted_contribs.copy()

            if 'Residual' in anova_table.index:
                pie_factors.append('Residual')
                pie_contribs.append(anova_table.loc['Residual', 'contribution'])

            # 如果贡献率小的太多，将它们合并为"其他"
            threshold = 3.0  # 3%以下的贡献率合并为"其他"
            other_sum = 0
            final_factors = []
            final_contribs = []

            for i, factor in enumerate(pie_factors):
                if pie_contribs[i] >= threshold or factor == 'Residual':
                    final_factors.append(factor)
                    final_contribs.append(pie_contribs[i])
                else:
                    other_sum += pie_contribs[i]

            if other_sum > 0:
                final_factors.append('其他')
                final_contribs.append(other_sum)

            # 生成饼图
            plt.pie(final_contribs, labels=final_factors, autopct='%1.1f%%',
                    startangle=90, shadow=False, explode=[0.05] * len(final_factors))
            plt.axis('equal')  # 保持圆形
            plt.title(f'{response_name}的因素贡献率', fontsize=14)

            # 保存饼图
            pie_filename = safe_filename(f'anova_pie_{response_name}', 'png', include_timestamp=True)
            pie_filepath = os.path.join(save_dir, pie_filename)
            plt.savefig(pie_filepath, dpi=300)
            plt.close()
            print(f"ANOVA贡献率饼图已保存至: {pie_filepath}")

            return csv_filepath

        except Exception as e:
            print(f"创建ANOVA表时出错: {str(e)}")
            traceback.print_exc()
            return None



    def plot_correlation_matrix(self, data, variables):
        """
        绘制变量之间的相关性矩阵

        Parameters:
        -----------
        data : pandas.DataFrame
            包含所有变量的数据框
        variables : list
            要包含在相关性矩阵中的变量名列表
        """
        try:
            # 确保目录存在
            save_dir = os.path.join(self.base_path, 'analysis_plots')
            os.makedirs(save_dir, exist_ok=True)

            # 计算相关性矩阵
            correlation = data[variables].corr()

            # 绘制相关性矩阵热图
            plt.figure(figsize=(10, 8))

            # 导入seaborn以获取更好的热图
            try:
                mask = np.triu(np.ones_like(correlation, dtype=bool))  # 上三角掩码
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                            annot=True, fmt=".2f", square=True, linewidths=.5)
            except ImportError:
                # 如果没有seaborn，使用matplotlib
                im = plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
                plt.colorbar(im)

                # 添加文本注释
                for i in range(len(correlation)):
                    for j in range(len(correlation)):
                        if i <= j:  # 只显示下三角部分
                            continue
                        text = plt.text(j, i, f'{correlation.iloc[i, j]:.2f}',
                                        ha="center", va="center", color="black")

            # 设置坐标轴标签
            plt.xticks(range(len(variables)), variables, rotation=45, ha='right')
            plt.yticks(range(len(variables)), variables)
            plt.title('变量相关性矩阵', fontsize=14)
            plt.tight_layout()

            # 保存图表
            filename = safe_filename(f'correlation_matrix', 'png', include_timestamp=True)
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()
            print(f"相关性矩阵图已保存至: {filepath}")

            return filepath

        except Exception as e:
            print(f"创建相关性矩阵图时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def plot_response_surface(self, data, var1, var2, response_name):
        """
        绘制响应曲面图，显示两个变量对响应的联合影响

        Parameters:
        -----------
        data : pandas.DataFrame
            包含所有变量的数据框
        var1 : str
            第一个设计变量名称
        var2 : str
            第二个设计变量名称
        response_name : str
            响应变量名称
        """
        try:
            # 确保目录存在
            save_dir = os.path.join(self.base_path, 'analysis_plots')
            os.makedirs(save_dir, exist_ok=True)


            # 提取数据
            x = data[var1].values
            y = data[var2].values
            z = data[response_name].values

            # 创建网格
            xi = np.linspace(min(x), max(x), 100)
            yi = np.linspace(min(y), max(y), 100)
            xi_grid, yi_grid = np.meshgrid(xi, yi)

            # 插值到网格上
            zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')

            # 创建3D图
            fig = plt.figure(figsize=(12, 10))

            # 3D曲面图
            ax1 = fig.add_subplot(221, projection='3d')
            surf = ax1.plot_surface(xi_grid, yi_grid, zi, cmap='viridis', alpha=0.8,
                                    linewidth=0, antialiased=True)
            fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
            ax1.set_xlabel(var1, fontsize=10)
            ax1.set_ylabel(var2, fontsize=10)
            ax1.set_zlabel(response_name, fontsize=10)
            ax1.set_title('响应曲面', fontsize=12)

            # 等高线图
            ax2 = fig.add_subplot(222)
            contour = ax2.contour(xi_grid, yi_grid, zi, 10, colors='k')
            ax2.clabel(contour, inline=1, fontsize=8)
            filled_c = ax2.contourf(xi_grid, yi_grid, zi, 100, cmap='viridis')
            fig.colorbar(filled_c, ax=ax2)
            ax2.scatter(x, y, c='r', marker='o', s=30, label='样本点')
            ax2.set_xlabel(var1, fontsize=10)
            ax2.set_ylabel(var2, fontsize=10)
            ax2.set_title('等高线图', fontsize=12)
            ax2.legend()

            # 按第一个变量分组的散点图
            ax3 = fig.add_subplot(223)
            ax3.scatter(x, z, c='b', alpha=0.6)
            ax3.set_xlabel(var1, fontsize=10)
            ax3.set_ylabel(response_name, fontsize=10)
            ax3.set_title(f'{var1}与{response_name}关系', fontsize=12)

            # 按第二个变量分组的散点图
            ax4 = fig.add_subplot(224)
            ax4.scatter(y, z, c='g', alpha=0.6)
            ax4.set_xlabel(var2, fontsize=10)
            ax4.set_ylabel(response_name, fontsize=10)
            ax4.set_title(f'{var2}与{response_name}关系', fontsize=12)

            plt.tight_layout()

            # 保存图表
            filename = safe_filename(f'response_surface_{var1}_{var2}_{response_name}', 'png', include_timestamp=True)
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()
            print(f"响应曲面图已保存至: {filepath}")

            return filepath

        except Exception as e:
            print(f"创建响应曲面图时出错: {str(e)}")
            import traceback
            traceback.print_exc()


    def plot_pareto_front(self, objectives, iteration):
        """绘制帕累托前沿"""
        try:
            if objectives.shape[0] == 0:
                print("警告: 帕累托前沿为空，跳过绘图")
                return

            # 创建保存目录
            save_dir = os.path.join(self.base_path, 'optimization_plots')
            os.makedirs(save_dir, exist_ok=True)

            # 时间戳现在由safe_filename处理，不需要单独生成
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            n_obj = objectives.shape[1]

            # 绘制2D的帕累托前沿图
            if n_obj >= 2:
                for i in range(n_obj):
                    for j in range(i + 1, n_obj):
                        plt.figure(figsize=(10, 8))
                        plt.scatter(objectives[:, i], objectives[:, j],
                                    c='blue', s=50, alpha=0.7, edgecolors='black')

                        plt.title(f'帕累托前沿: {self.obj_names[i]} vs {self.obj_names[j]} - 迭代 {iteration}',
                                  fontsize=14)
                        plt.xlabel(self.obj_names[i], fontsize=12)
                        plt.ylabel(self.obj_names[j], fontsize=12)
                        plt.grid(True, linestyle='--', alpha=0.6)

                        # 保存图片
                        filename = safe_filename(f'pareto_front_2d_{self.obj_names[i]}_{self.obj_names[j]}', 'png', include_timestamp=True, iteration=iteration)
                        filepath = os.path.join(save_dir, filename)
                        plt.tight_layout()
                        plt.savefig(filepath, dpi=300)
                        plt.close()
                        print(f"2D帕累托前沿图已保存至: {filepath}")

            # 绘制3D的帕累托前沿图
            if n_obj >= 3:
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')

                scatter = ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                                     c='blue', s=50, alpha=0.7, edgecolors='black')

                ax.set_title(f'3D帕累托前沿 - 迭代 {iteration}', fontsize=14)
                ax.set_xlabel(self.obj_names[0], fontsize=12)
                ax.set_ylabel(self.obj_names[1], fontsize=12)
                ax.set_zlabel(self.obj_names[2], fontsize=12)

                # 保存图片
                filename = safe_filename('pareto_front_3d', 'png', include_timestamp=True, iteration=iteration)
                filepath = os.path.join(save_dir, filename)
                plt.tight_layout()
                plt.savefig(filepath, dpi=300)
                plt.close()
                print(f"3D帕累托前沿图已保存至: {filepath}")

        except Exception as e:
            print(f"绘制帕累托前沿时出错: {str(e)}")

    def plot_pareto_front_analysis(self, data, obj1, obj2):
        """
        绘制二维帕累托前沿图(用于最终分析)

        Parameters:
        -----------
        data : pandas.DataFrame
            包含所有目标值的数据框
        obj1 : str
            第一个目标函数名称
        obj2 : str
            第二个目标函数名称
        """
        try:
            # 确保目录存在
            save_dir = os.path.join(self.base_path, 'analysis_plots')
            os.makedirs(save_dir, exist_ok=True)

            # 提取数据
            x = data[obj1].values
            y = data[obj2].values
            sample_type = data['Sample_Type'] if 'Sample_Type' in data.columns else ['unknown'] * len(x)

            # 创建图像
            plt.figure(figsize=(10, 8))

            # 根据样本类型分类绘制
            colors = {'DOE': 'blue', 'Optimization': 'red', 'unknown': 'gray'}
            markers = {'DOE': 'o', 'Optimization': '^', 'unknown': 's'}

            for stype in set(sample_type):
                mask = [s == stype for s in sample_type]
                plt.scatter(
                    [x[i] for i in range(len(x)) if mask[i]],
                    [y[i] for i in range(len(y)) if mask[i]],
                    color=colors.get(stype, 'gray'),
                    marker=markers.get(stype, 'o'),
                    label=stype,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.5
                )

            # 识别帕累托最优解
            pareto_points = self._identify_pareto_points(x, y)
            pareto_x = [x[i] for i in range(len(x)) if pareto_points[i]]
            pareto_y = [y[i] for i in range(len(y)) if pareto_points[i]]

            # 连接帕累托前沿点
            if len(pareto_x) > 1:
                # 按第一个目标排序
                sorted_indices = np.argsort(pareto_x)
                pareto_x_sorted = [pareto_x[i] for i in sorted_indices]
                pareto_y_sorted = [pareto_y[i] for i in sorted_indices]

                plt.plot(pareto_x_sorted, pareto_y_sorted, 'k--', linewidth=2, label='帕累托前沿')

            plt.xlabel(obj1, fontsize=12)
            plt.ylabel(obj2, fontsize=12)
            plt.title(f'{obj1} vs {obj2} 帕累托前沿分析', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            # 保存图表
            filename = safe_filename(f'pareto_analysis_{obj1}_{obj2}', 'png', include_timestamp=True)
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()
            print(f"帕累托前沿分析图已保存至: {filepath}")

            return filepath

        except Exception as e:
            print(f"创建帕累托前沿分析图时出错: {str(e)}")
            traceback.print_exc()
            return None

    def plot_3d_pareto_front_analysis(self, data, obj1, obj2, obj3):
        """
        绘制三维帕累托前沿图(用于最终分析)

        Parameters:
        -----------
        data : pandas.DataFrame
            包含所有目标值的数据框
        obj1 : str
            第一个目标函数名称
        obj2 : str
            第二个目标函数名称
        obj3 : str
            第三个目标函数名称
        """
        try:
            # 确保目录存在
            save_dir = os.path.join(self.base_path, 'analysis_plots')
            os.makedirs(save_dir, exist_ok=True)

            # 提取数据
            x = data[obj1].values
            y = data[obj2].values
            z = data[obj3].values
            sample_type = data['Sample_Type'] if 'Sample_Type' in data.columns else ['unknown'] * len(x)

            # 创建3D图像
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # 根据样本类型分类绘制
            colors = {'DOE': 'blue', 'Optimization': 'red', 'unknown': 'gray'}
            markers = {'DOE': 'o', 'Optimization': '^', 'unknown': 's'}

            for stype in set(sample_type):
                mask = [s == stype for s in sample_type]
                ax.scatter(
                    [x[i] for i in range(len(x)) if mask[i]],
                    [y[i] for i in range(len(y)) if mask[i]],
                    [z[i] for i in range(len(z)) if mask[i]],
                    color=colors.get(stype, 'gray'),
                    marker=markers.get(stype, 'o'),
                    label=stype,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.5
                )

            # 识别帕累托最优点
            pareto_points = self._identify_3d_pareto_points(x, y, z)
            pareto_x = [x[i] for i in range(len(x)) if pareto_points[i]]
            pareto_y = [y[i] for i in range(len(y)) if pareto_points[i]]
            pareto_z = [z[i] for i in range(len(z)) if pareto_points[i]]

            # 标记帕累托最优点
            if pareto_x:
                ax.scatter(pareto_x, pareto_y, pareto_z,
                           color='lime', marker='*', s=100,
                           label='帕累托最优点',
                           edgecolors='black', linewidths=0.5)

            ax.set_xlabel(obj1, fontsize=10)
            ax.set_ylabel(obj2, fontsize=10)
            ax.set_zlabel(obj3, fontsize=10)
            ax.set_title(f'三维帕累托前沿分析: {obj1} vs {obj2} vs {obj3}', fontsize=14)
            ax.legend()

            # 调整视角
            ax.view_init(elev=30, azim=45)

            # 保存图表
            filename = safe_filename(f'3d_pareto_analysis_{obj1}_{obj2}_{obj3}', 'png', include_timestamp=True)
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300)

            # 保存不同角度的视图
            for angle in [0, 90, 180, 270]:
                ax.view_init(elev=30, azim=angle)
                angle_filename = safe_filename(f'3d_pareto_analysis_{obj1}_{obj2}_{obj3}_angle_{angle}', 'png')
                angle_filepath = os.path.join(save_dir, angle_filename)
                plt.savefig(angle_filepath, dpi=300)

            plt.close()
            print(f"三维帕累托前沿分析图已保存至: {filepath} (并包含4个不同角度视图)")

            return filepath

        except Exception as e:
            print(f"创建三维帕累托前沿分析图时出错: {str(e)}")
            traceback.print_exc()
            return None

    def _identify_pareto_points(self, costs1, costs2):
        """
        识别二维帕累托最优点
        假设目标是最小化两个成本函数

        Parameters:
        -----------
        costs1 : list or numpy.ndarray
            第一个成本函数值列表
        costs2 : list or numpy.ndarray
            第二个成本函数值列表

        Returns:
        --------
        list of bool: 长度等于输入列表，True表示该点是帕累托最优的
        """
        costs = list(zip(costs1, costs2))
        is_pareto = [True] * len(costs)

        for i, cost_i in enumerate(costs):
            if not is_pareto[i]:
                continue

            for j, cost_j in enumerate(costs):
                if i == j:
                    continue

                # 如果存在一个点在所有维度上都不劣于当前点，且至少一个维度上严格优于当前点
                # 则当前点不是帕累托最优的
                if all(cost_j[k] <= cost_i[k] for k in range(len(cost_i))) and \
                        any(cost_j[k] < cost_i[k] for k in range(len(cost_i))):
                    is_pareto[i] = False
                    break

        return is_pareto

    def _identify_3d_pareto_points(self, costs1, costs2, costs3):
        """
        识别三维帕累托最优点
        假设目标是最小化三个成本函数

        Parameters:
        -----------
        costs1, costs2, costs3 : list or numpy.ndarray
            三个成本函数值列表

        Returns:
        --------
        list of bool: 长度等于输入列表，True表示该点是帕累托最优的
        """
        costs = list(zip(costs1, costs2, costs3))
        is_pareto = [True] * len(costs)

        for i, cost_i in enumerate(costs):
            if not is_pareto[i]:
                continue

            for j, cost_j in enumerate(costs):
                if i == j:
                    continue

                # 如果存在一个点在所有维度上都不劣于当前点，且至少一个维度上严格优于当前点
                # 则当前点不是帕累托最优的
                if all(cost_j[k] <= cost_i[k] for k in range(len(cost_i))) and \
                        any(cost_j[k] < cost_i[k] for k in range(len(cost_i))):
                    is_pareto[i] = False
                    break

        return is_pareto




    def plot_optimization_history(self, history, include_initial_doe=True, doe_file=None):
        """
        绘制优化历史，包含初始DOE数据，并清晰区分DOE点和优化点

        Parameters:
        -----------
        history : dict
            优化历史数据字典
        include_initial_doe : bool
            是否包含初始DOE数据
        doe_file : str
            DOE数据文件路径
        """
        try:
            # 创建保存目录
            save_dir = os.path.join(self.base_path, 'optimization_plots')
            os.makedirs(save_dir, exist_ok=True)

            # 设置matplotlib支持中文
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False

            # 加载初始DOE数据
            initial_doe_data = None
            n_doe_samples = 0
            if include_initial_doe and doe_file is not None and os.path.exists(doe_file):
                try:
                    initial_doe_data = pd.read_csv(doe_file, delimiter='\t')
                    n_doe_samples = len(initial_doe_data)
                    print(f"成功加载初始DOE数据: {doe_file}, 数据点数: {n_doe_samples}")
                except Exception as e:
                    print(f"加载初始DOE数据时出错: {str(e)}")

            # 为每个目标函数单独绘制优化历史（增强版-区分DOE和优化点）
            for obj_name in self.obj_names:
                if obj_name not in history:
                    print(f"警告: {obj_name}的历史数据不完整，跳过绘图")
                    continue

                # 创建图形
                plt.figure(figsize=(12, 7))

                # 收集DOE数据
                doe_x = []
                doe_y = []
                if initial_doe_data is not None and obj_name in initial_doe_data.columns:
                    doe_values = initial_doe_data[obj_name].values
                    doe_x = list(range(len(doe_values)))
                    doe_y = doe_values.tolist()

                    # 绘制DOE散点
                    plt.scatter(doe_x, doe_y, color='green', s=50, marker='o',
                                label='初始DOE样本', zorder=5)

                # 收集优化点数据
                opt_start = len(doe_x)
                opt_x = list(range(opt_start, opt_start + len(history[obj_name])))
                opt_y = history[obj_name]

                # 绘制优化点
                plt.scatter(opt_x, opt_y, color='red', s=80, marker='*',
                            label='优化样本', zorder=6)

                # 绘制连接线（完整历史）
                all_x = doe_x + opt_x
                all_y = doe_y + list(opt_y)
                plt.plot(all_x, all_y, 'b-', alpha=0.5, label='完整轨迹')

                # 如果有DOE数据，添加分割线
                if len(doe_x) > 0:
                    plt.axvline(x=len(doe_x) - 0.5, color='red', linestyle='-', linewidth=1.5,
                                label='优化开始')

                    # 添加DOE和优化区域标签
                    plt.text(len(doe_x) / 2, min(all_y) + 0.9 * (max(all_y) - min(all_y)),
                             "DOE阶段", color='green', fontsize=12, ha='center',
                             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

                    plt.text(len(doe_x) + len(opt_x) / 2, min(all_y) + 0.9 * (max(all_y) - min(all_y)),
                             "优化阶段", color='red', fontsize=12, ha='center',
                             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

                # 添加点数量标注
                plt.annotate(f'DOE点数: {len(doe_x)}', xy=(0.02, 0.97), xycoords='axes fraction',
                             bbox=dict(facecolor='white', alpha=0.7))
                plt.annotate(f'优化点数: {len(opt_x)}', xy=(0.02, 0.92), xycoords='axes fraction',
                             bbox=dict(facecolor='white', alpha=0.7))

                # 设置图表属性
                plt.title(f'{obj_name}优化历史', fontsize=16)
                plt.xlabel('迭代步骤', fontsize=14)
                plt.ylabel(f'{obj_name}值', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)

                # 确保x轴只显示整数刻度

                ax = plt.gca()
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                # 设置x轴范围
                plt.xlim(-0.5, max(all_x) + 0.5 if all_x else 0.5)

                # 添加图例
                plt.legend(loc='best', fontsize=10)

                # 保存图片
                filename = safe_filename(f'optimization_history_{obj_name}', 'png', include_timestamp=True)
                filepath = os.path.join(save_dir, filename)
                plt.tight_layout()
                plt.savefig(filepath, dpi=300)
                plt.close()
                print(f"{obj_name}优化历史图已保存至: {filepath}")

            # 绘制所有目标函数的联合图（使用独立坐标轴而非归一化）
            plt.figure(figsize=(14, 8))

            colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
            markers = ['o', 's', '^', 'd', '*', 'x']

            # 创建主坐标轴
            ax = plt.gca()
            axes = [ax]  # 存储所有坐标轴的列表
            lines = []  # 存储所有线的列表

            # 为每个目标函数创建独立的y轴
            for i, obj_name in enumerate(self.obj_names):
                if obj_name not in history:
                    continue

                if i > 0:
                    # 为第二个及后续目标创建独立y轴
                    ax_new = ax.twinx()
                    # 偏移第三个及后续y轴以避免重叠
                    if i >= 2:
                        offset = 60 * (i - 1)
                        ax_new.spines['right'].set_position(('outward', offset))
                    axes.append(ax_new)

                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                current_ax = axes[i]

                # 收集DOE数据
                doe_x = []
                doe_y = []
                if initial_doe_data is not None and obj_name in initial_doe_data.columns:
                    doe_values = initial_doe_data[obj_name].values
                    doe_x = list(range(len(doe_values)))
                    doe_y = doe_values.tolist()

                    # 绘制DOE点
                    current_ax.scatter(doe_x, doe_y, color=color, s=30, alpha=0.6, marker='o')

                # 收集优化点数据
                opt_start = len(doe_x)
                opt_x = list(range(opt_start, opt_start + len(history[obj_name])))
                opt_y = history[obj_name]

                # 绘制优化点和连接线
                line = current_ax.plot(opt_x, opt_y, f'-{marker}',
                                       color=color, linewidth=2, markersize=5,
                                       label=f'{obj_name}')[0]
                lines.append(line)

                # 设置标签
                current_ax.set_ylabel(f'{obj_name}', color=color, fontsize=12)
                current_ax.tick_params(axis='y', labelcolor=color)

            # 如果有DOE数据，添加分割线
            if n_doe_samples > 0:
                ax.axvline(x=n_doe_samples - 0.5, color='red', linestyle='-', linewidth=1.5)

                # 添加阶段标签
                y_pos = ax.get_ylim()[0] + 0.9 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                ax.text(n_doe_samples / 2, y_pos, "DOE阶段",
                        color='black', fontsize=12, ha='center',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

                ax.text(n_doe_samples + len(opt_x) / 2, y_pos, "优化阶段",
                        color='black', fontsize=12, ha='center',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

            # 设置图表属性
            plt.title('多目标优化历史', fontsize=16)
            plt.xlabel('迭代步骤', fontsize=14)

            # 确保x轴只显示整数刻度
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # 创建合并的图例
            ax.legend(lines, [self.obj_names[i] for i in range(len(lines))],
                      loc='best', fontsize=10)

            # 保存联合图
            filename = safe_filename('optimization_history_all_objectives', 'png', include_timestamp=True)
            filepath = os.path.join(save_dir, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=600)
            plt.close()
            print(f"联合目标优化历史图已保存至: {filepath}")

        except Exception as e:
            print(f"绘制优化历史时出错: {str(e)}")
            traceback.print_exc()


def plot_objective_history_with_markers(self, obj_name, history, initial_doe_data=None):
    """为单个目标绘制优化历史图，明确区分DOE点和优化点"""
    try:
        # 创建保存目录
        save_dir = os.path.join(self.base_path, 'optimization_plots')
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(12, 7))

        # 处理DOE数据
        doe_x = []
        doe_y = []
        if initial_doe_data is not None and obj_name in initial_doe_data.columns:
            doe_values = initial_doe_data[obj_name].values
            doe_x = list(range(len(doe_values)))
            doe_y = doe_values

            # 绘制DOE点，使用圆形标记
            plt.scatter(doe_x, doe_y, color='green', s=50, marker='o',
                        label='初始DOE样本', zorder=5)

        # 处理优化数据
        opt_x = list(range(len(doe_x), len(doe_x) + len(history[obj_name])))
        opt_y = history[obj_name]

        # 绘制优化点，使用星形标记，确保所有点都可见
        plt.scatter(opt_x, opt_y, color='red', s=80, marker='*',
                    label='优化样本', zorder=6)

        # 绘制完整历史曲线
        full_x = doe_x + opt_x
        full_y = doe_y.tolist() + opt_y.tolist() if len(doe_y) > 0 else opt_y.tolist()
        plt.plot(full_x, full_y, 'b-', alpha=0.5, label='完整轨迹')

        # 如果有DOE数据，添加分割线
        if len(doe_x) > 0:
            plt.axvline(x=len(doe_x) - 0.5, color='red', linestyle='-', linewidth=1.5,
                        label='优化开始')

            plt.text(len(doe_x) / 2, min(full_y) + 0.9 * (max(full_y) - min(full_y)),
                     "DOE阶段", color='green', fontsize=12, ha='center',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

            plt.text(len(doe_x) + len(opt_x) / 2, min(full_y) + 0.9 * (max(full_y) - min(full_y)),
                     "优化阶段", color='red', fontsize=12, ha='center',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

        # 设置图表属性
        plt.title(f'{obj_name}优化历史', fontsize=16)
        plt.xlabel('迭代步骤', fontsize=14)
        plt.ylabel(f'{obj_name}值', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 确保x轴只显示整数刻度
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # 添加数据点数量信息
        plt.annotate(f'DOE点数量: {len(doe_x)}', xy=(0.02, 0.97), xycoords='axes fraction',
                     bbox=dict(facecolor='white', alpha=0.7))
        plt.annotate(f'优化点数量: {len(opt_x)}', xy=(0.02, 0.92), xycoords='axes fraction',
                     bbox=dict(facecolor='white', alpha=0.7))

        plt.legend(loc='best', fontsize=10)

        # 保存图片
        filename = safe_filename(f'optimization_history_{obj_name}', 'png', include_timestamp=True)
        filepath = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"{obj_name}优化历史图(含标记)已保存至: {filepath}")

    except Exception as e:
        print(f"绘制目标优化历史图(含标记)时出错: {str(e)}")
        traceback.print_exc()


def use_surrogate_model_example(self, models, var_names):
    """展示如何使用训练好的代理模型进行预测"""
    try:
        print("\n=== 代理模型使用示例 ===")

        # 创建示例输入点
        n_points = 5

        # 使用均匀分布在变量范围内生成示例点
        example_points = np.array([
            [300, 1000, 20],  # 示例点1
            [400, 1200, 25],  # 示例点2
            [500, 1500, 30],  # 示例点3
            [350, 1100, 15],  # 示例点4
            [450, 1300, 35]  # 示例点5
        ])

        # 使用代理模型进行预测
        predictions = np.zeros((n_points, len(self.obj_names)))

        for i, obj_name in enumerate(self.obj_names):
            if obj_name in models:
                predictions[:, i] = models[obj_name].predict(example_points)

        # 显示预测结果
        print("\n预测结果:")
        print("=" * 50)
        print(f"{'设计变量':<30} | {'预测的目标值'}")
        print("-" * 50)

        for i in range(n_points):
            vars_str = ", ".join(
                [f"{var_names[j]}={example_points[i, j]:.2f}" for j in range(len(var_names))])
            preds_str = ", ".join(
                [f"{self.obj_names[j]}={predictions[i, j]:.6f}" for j in range(len(self.obj_names))])
            print(f"{vars_str:<30} | {preds_str}")

        print("=" * 50)

        # 保存预测结果
        results_file = os.path.join(self.base_path, 'surrogate_models', 'example_predictions.csv')
        results_df = pd.DataFrame()

        for j in range(len(var_names)):
            results_df[var_names[j]] = example_points[:, j]

        for j in range(len(self.obj_names)):
            results_df[self.obj_names[j]] = predictions[:, j]

        results_df.to_csv(results_file, index=False)
        print(f"\n示例预测结果已保存至: {results_file}")

        return results_df

    except Exception as e:
        print(f"代理模型使用示例出错: {str(e)}")
        return None


class SurrogateModelManager:
    def __init__(self, base_path, n_folds=10):
        self.base_path = base_path
        self.n_folds = n_folds
        self.models = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建模型保存目录
        os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)

    def cross_validate_model(self, X, y, objective_name, model_class=None):
        """执行k折交叉验证，增加数据检查和稳健性"""
        try:
            # 检测并处理异常值
            outliers = self._check_outliers(X, y)
            if outliers:
                print(f"警告: 检测到{len(outliers)}个可能的异常值")

            # 确保y不含异常值
            y_mean = np.mean(y)
            y_std = np.std(y)
            y_min = np.min(y)
            y_max = np.max(y)
            print(f"目标值 {objective_name} 统计信息:")
            print(f"  均值: {y_mean:.6f}, 标准差: {y_std:.6f}")
            print(f"  最小值: {y_min:.6f}, 最大值: {y_max:.6f}")

            # 异常值处理 - 可选
            is_outlier = np.abs(y - y_mean) > 3 * y_std
            if np.any(is_outlier):
                print(f"警告: 检测到{np.sum(is_outlier)}个目标值异常点")
                print("这些异常值可能导致R2计算异常")
                # 可以选择替换或去除异常值

            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            scores = []
            predictions = []

            # 如果没有指定模型类，使用默认的KrigingModel
            if model_class is None:
                model_class = KrigingModel

            for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # 检查训练集和测试集
                if len(np.unique(y_test)) <= 1:
                    print(f"警告: 折 {fold} 测试集中目标值没有变化，这可能导致R2计算问题")

                # 创建并训练模型
                model = model_class(normalize=True)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # 安全的R2计算
                r2 = self._safe_r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                max_err = max_error(y_test, y_pred)

                # R2值合理性检查
                if r2 > 1.0:
                    print(f"警告: 折 {fold} 的R²值异常偏高 ({r2:.4f})")
                    r2 = min(r2, 1.0)  # 限制R2最大为1
                    print(f"已将R²调整为: {r2:.4f}")
                elif r2 < -1.0:
                    print(f"警告: 折 {fold} 的R²值异常偏低 ({r2:.4f})")
                    print("检查数据:")
                    print(f"  测试集大小: {len(y_test)}")
                    print(f"  测试集目标均值: {np.mean(y_test):.6f}, 标准差: {np.std(y_test):.6f}")
                    print(f"  预测值均值: {np.mean(y_pred):.6f}, 标准差: {np.std(y_pred):.6f}")
                    self._investigate_poor_fit(y_test, y_pred, fold, objective_name)

                scores.append({
                    'fold': fold,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'max_error': max_err
                })
                predictions.append(y_pred)

                print(f"折 {fold}/{self.n_folds}:")
                print(f"  R²: {r2:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE: {mae:.4f}")
                print(f"  最大误差: {max_err:.4f}")

            # 计算平均分数
            avg_r2 = np.mean([s['r2'] for s in scores])
            avg_rmse = np.mean([s['rmse'] for s in scores])
            avg_mae = np.mean([s['mae'] for s in scores])

            print(f"\n{self.n_folds}折交叉验证平均结果:")
            print(f"平均 R²: {avg_r2:.4f}")
            print(f"平均 RMSE: {avg_rmse:.4f}")
            print(f"平均 MAE: {avg_mae:.4f}")

            # 保存交叉验证结果
            self.save_cv_results(scores, objective_name)

            return scores, predictions

        except Exception as e:
            print(f"交叉验证出错: {str(e)}")
            traceback.print_exc()
            return [], []

    def _safe_r2_score(self, y_true, y_pred):
        """安全计算R2分数，处理极端情况"""
        try:
            # 检查数据是否符合要求
            if len(y_true) <= 1:
                print("警告: 样本数量不足，无法计算有效的R²")
                return 0.0

            if np.allclose(y_true, y_true[0], rtol=1e-10, atol=1e-10):
                print("警告: 真实值几乎没有变化，R²可能不稳定")
                return 0.0

            # 计算R2值
            r2 = r2_score(y_true, y_pred)

            # 处理异常值
            if not np.isfinite(r2) or r2 > 1.0:
                print(f"警告: R²值异常 ({r2})")
                # 手动计算R2
                y_mean = np.mean(y_true)
                ss_total = np.sum((y_true - y_mean) ** 2)
                ss_residual = np.sum((y_true - y_pred) ** 2)

                if ss_total < 1e-10:  # 如果分母接近0
                    print("警告: 总平方和接近0，使用替代计算")
                    return 1.0 - ss_residual / max(ss_total, 1e-10)

                manual_r2 = 1.0 - ss_residual / ss_total
                print(f"手动计算的R²: {manual_r2:.4f}")
                return min(manual_r2, 1.0)  # 限制最大值为1

            return min(r2, 1.0)  # 确保R2不超过1

        except Exception as e:
            print(f"R²计算错误: {str(e)}")
            return 0.0

    def _investigate_poor_fit(self, y_true, y_pred, fold, objective_name):
        """调查不良拟合情况"""
        try:
            # 计算残差
            residuals = y_true - y_pred

            # 找出最大残差点
            max_resid_idx = np.argmax(np.abs(residuals))
            print(f"最大残差点:")
            print(f"  真实值: {y_true[max_resid_idx]:.6f}")
            print(f"  预测值: {y_pred[max_resid_idx]:.6f}")
            print(f"  残差: {residuals[max_resid_idx]:.6f}")

            # 残差分析
            plt.figure(figsize=(12, 5))

            # 残差散点图
            plt.subplot(121)
            plt.scatter(y_pred, residuals)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('残差散点图')
            plt.xlabel('预测值')
            plt.ylabel('残差')

            # 残差分布直方图
            plt.subplot(122)
            plt.hist(residuals, bins=10)
            plt.title('残差分布')
            plt.xlabel('残差')
            plt.ylabel('频率')

            # 保存图像
            save_dir = os.path.join(self.base_path, 'model_validation')
            os.makedirs(save_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{objective_name}_fold_{fold}_residual_analysis.png'))
            plt.close()

            print(f"残差分析图已保存")

        except Exception as e:
            print(f"调查拟合问题时出错: {str(e)}")

    def _check_outliers(self, X, y, threshold=3.0):
        """检测并返回异常值"""
        # 标准化数据
        X_norm = (X - np.mean(X, axis=0)) / np.maximum(np.std(X, axis=0), 1e-10)
        y_norm = (y - np.mean(y)) / max(np.std(y), 1e-10)

        # 计算马氏距离（简化版）
        outliers = []
        for i in range(len(X)):
            # 对于输入特征的距离
            x_dist = np.sum(X_norm[i] ** 2)
            # 对于输出值的距离
            y_dist = y_norm[i] ** 2
            # 总距离
            dist = np.sqrt(x_dist + y_dist)
            if dist > threshold:
                outliers.append((i, dist))

        return sorted(outliers, key=lambda x: -x[1])  # 按距离降序排序

    def save_cv_results(self, scores, objective_name):
        """保存交叉验证结果"""
        try:
            save_dir = os.path.join(self.base_path, 'model_validation')
            os.makedirs(save_dir, exist_ok=True)

            # 保存为CSV
            df = pd.DataFrame(scores)
            csv_file = os.path.join(save_dir, safe_filename(f'{objective_name}_cv_results', 'csv', include_timestamp=True))
            df.to_csv(csv_file, index=False)

            # 创建条形图
            plt.figure(figsize=(12, 6))
            folds = [s['fold'] for s in scores]
            r2_values = [s['r2'] for s in scores]
            rmse_values = [s['rmse'] for s in scores]

            x = np.arange(len(folds))
            width = 0.35

            ax1 = plt.subplot(121)
            ax1.bar(x, r2_values, width, label='R²')
            ax1.set_xlabel('折数')
            ax1.set_ylabel('R² 分数')
            ax1.set_title(f'{objective_name} - R² 分数')
            ax1.set_xticks(x)
            ax1.set_xticklabels(folds)

            ax2 = plt.subplot(122)
            ax2.bar(x, rmse_values, width, color='orange', label='RMSE')
            ax2.set_xlabel('折数')
            ax2.set_ylabel('RMSE')
            ax2.set_title(f'{objective_name} - RMSE')
            ax2.set_xticks(x)
            ax2.set_xticklabels(folds)

            plt.tight_layout()
            plt_file = os.path.join(save_dir, safe_filename(f'{objective_name}_cv_results', 'png', include_timestamp=True))
            plt.savefig(plt_file, dpi=300)
            plt.close()

            print(f"交叉验证结果已保存至: {csv_file} 和 {plt_file}")

        except Exception as e:
            print(f"保存交叉验证结果时出错: {str(e)}")

    def save_model(self, model, objective_name):
        """保存训练好的代理模型"""
        try:
            model_dir = os.path.join(self.base_path, 'models')
            os.makedirs(model_dir, exist_ok=True)

            filename = safe_filename(f'surrogate_model_{objective_name}', 'pkl', include_timestamp=True)
            filepath = os.path.join(model_dir, filename)

            with open(filepath, 'wb') as f:
                pickle.dump(model, f)

            self.models[objective_name] = {
                'model': model,
                'filename': filepath,
                'timestamp': self.timestamp
            }

            print(f"{objective_name}代理模型已保存至: {filepath}")

            # 保存模型参数
            if hasattr(model, 'kernel_'):
                params_file = os.path.join(model_dir, safe_filename(f'surrogate_model_{objective_name}_params', 'txt', include_timestamp=True))
                with open(params_file, 'w') as f:
                    f.write(f"模型: {type(model).__name__}\n")
                    f.write(f"目标: {objective_name}\n")
                    f.write(f"时间: {self.timestamp}\n\n")
                    f.write(f"超参数: {model.get_params()}\n")
                    f.write(f"核函数: {model.kernel_}\n")
                    if hasattr(model, 'theta_'):
                        f.write(f"theta: {model.theta_}\n")

                print(f"{objective_name}代理模型参数已保存至: {params_file}")

            return filepath

        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            return None

    def load_model(self, objective_name, timestamp=None):
        """加载保存的代理模型"""
        try:
            model_dir = os.path.join(self.base_path, 'models')

            if timestamp is None:
                # 如果没有指定时间戳，使用最新的模型
                files = glob.glob(os.path.join(model_dir, f'surrogate_model_{objective_name}_*.pkl'))
                if not files:
                    raise FileNotFoundError(f"找不到{objective_name}的代理模型文件")

                latest_file = max(files, key=os.path.getctime)
                filepath = latest_file
            else:
                filepath = os.path.join(model_dir, f'surrogate_model_{objective_name}_{timestamp}.pkl')

            print(f"加载模型: {filepath}")
            with open(filepath, 'rb') as f:
                model = pickle.load(f)

            self.models[objective_name] = {
                'model': model,
                'filename': filepath,
                'timestamp': timestamp or os.path.basename(filepath).split('_')[-1].split('.')[0]
            }

            return model

        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return None

    def predict_with_model(self, model_or_name, X):
        """使用代理模型进行预测"""
        try:
            if isinstance(model_or_name, str):
                if model_or_name in self.models:
                    model = self.models[model_or_name]['model']
                else:
                    model = self.load_model(model_or_name)
            else:
                model = model_or_name

            return model.predict(X)

        except Exception as e:
            print(f"预测时出错: {str(e)}")
            return None
