import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import traceback

# 设置中文字体
try:
    # 尝试设置微软雅黑
    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams['font.family'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("警告: 无法加载中文字体，图表中的中文可能无法正确显示")


class SurrogateValidator:
    def __init__(self, base_path=None, save_dir=None):
        """初始化代理模型验证器

        参数:
        ----
        base_path : str, 可选
            项目基础路径
        save_dir : str, 可选
            保存验证结果的目录
        """
        self.base_path = base_path or os.getcwd()
        self.save_dir = save_dir or os.path.join(self.base_path, "surrogate_validation")

        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)

        # 初始化验证历史记录 - 采用您原有的字典格式
        self.history = {}

    def validate_model(self, model, X_test, y_test, obj_name, iteration=None):
        """验证代理模型并记录结果

        参数:
        ----
        model : object
            训练好的代理模型
        X_test : np.ndarray
            测试数据的特征
        y_test : np.ndarray
            测试数据的目标值
        obj_name : str
            目标函数名称 ('Cl', 'Cd', 或 'Cm')
        iteration : int, 可选
            模型的迭代次数

        返回:
        ----
        dict : 包含各种验证指标的字典
        """
        try:
            # 使用模型进行预测
            y_pred = model.predict(X_test)

            # 计算各种指标
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            max_err = max_error(y_test, y_pred)

            # 计算相对误差
            rel_error = np.mean(np.abs((y_test - y_pred) / np.clip(np.abs(y_test), 1e-10, None))) * 100

            # 计算指标
            metrics = {
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'max_error': max_err,
                'rel_error': rel_error
            }

            # 添加到历史记录 - 使用字典格式
            if iteration not in self.history:
                self.history[iteration] = {}

            self.history[iteration][obj_name] = metrics

            return metrics

        except Exception as e:
            print(f"验证模型时出错: {str(e)}")

            traceback.print_exc()
            return None

    def plot_validation_metrics(self, iteration=None, save_dir=None):
        """绘制验证指标随时间的变化

        参数:
        ----
        iteration : int, 可选
            当前迭代次数，用于标记图表
        save_dir : str, 可选
            保存图表的目录，如果为None则使用self.save_dir
        """
        try:
            # 使用传入的save_dir或默认目录
            save_dir = save_dir or self.save_dir
            os.makedirs(save_dir, exist_ok=True)

            # 检查历史数据
            if not self.history:
                print("验证历史为空，无法绘图")
                return

            # 设置matplotlib参数
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False

            # 设置子图布局
            fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

            metrics = ['r2', 'rmse', 'rel_error']
            titles = ['R² 分数', 'RMSE', '相对误差 (%)']
            colors = {'Cl': 'blue', 'Cd': 'red', 'Cm': 'green'}

            # 获取所有迭代次数
            iterations = sorted(self.history.keys())

            for ax_idx, metric in enumerate(metrics):
                ax = axes[ax_idx]

                # 为每个目标绘制曲线
                for obj_name in ['Cl', 'Cd', 'Cm']:
                    values = []
                    iters = []

                    for iter_num in iterations:
                        if obj_name in self.history[iter_num]:
                            iters.append(iter_num)
                            values.append(self.history[iter_num][obj_name][metric])

                    if values:
                        # 绘制曲线
                        ax.plot(iters, values, marker='o', linestyle='-',
                                color=colors[obj_name], label=obj_name)

                        # 为每个点添加数值标签
                        for i, val in enumerate(values):
                            # 限制显示的小数位数，根据数值范围
                            if abs(val) > 10:
                                val_text = f"{val:.1f}"
                            elif abs(val) > 1:
                                val_text = f"{val:.2f}"
                            else:
                                val_text = f"{val:.3f}"

                            # 注意y轴位置，避免超出图表
                            ax.annotate(val_text, (iters[i], values[i]),
                                        textcoords="offset points",
                                        xytext=(0, 5),
                                        ha='center')

                # 设置标题和标签
                ax.set_title(titles[ax_idx], fontsize=14)
                ax.set_ylabel(metric, fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)

                # 确保所有数据点都在可见范围内
                if metric == 'r2':
                    y_min, y_max = ax.get_ylim()
                    ax.set_ylim([min(y_min, -0.1), max(y_max, 1.0)])

                # 图例位置 - 选择最佳位置
                if ax_idx == 0:  # R²图
                    ax.legend(loc='lower right')
                elif ax_idx == 1:  # RMSE图
                    ax.legend(loc='upper right')
                else:  # 相对误差图
                    ax.legend(loc='upper right')

            # 设置共享的x轴属性
            axes[-1].set_xlabel('迭代次数', fontsize=14)

            # 确保x轴只显示整数刻度
            from matplotlib.ticker import MaxNLocator
            axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))

            # 调整布局
            plt.tight_layout()

            # 保存图片
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'validation_metrics_{timestamp}.png'
            if iteration is not None:
                filename = f'validation_metrics_iter{iteration}_{timestamp}.png'

            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"验证指标图已保存至: {save_path}")

        except Exception as e:
            print(f"绘制验证指标图时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def plot_improved_validation_metrics(metrics_data, iteration, save_path=None):
        """
        绘制改进后的验证指标图表，解决以下问题：
        1. 添加明确的横坐标标签
        2. 处理异常值
        3. 避免数据标签重叠

        Parameters:
        -----------
        metrics_data : dict
            包含验证指标的字典，键为指标名称，值为目标函数对应的指标值列表
        iteration : int
            当前迭代次数
        save_path : str, optional
            图表保存路径
        """
        # 设置优雅的绘图风格
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('seaborn-whitegrid')  # 兼容不同版本的matplotlib

        # 获取指标名称和目标数量
        metrics = list(metrics_data.keys())
        n_objectives = len(next(iter(metrics_data.values())))

        # 创建子图，每个指标一个子图
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]  # 确保axes始终是列表

        # 为每个指标创建一个子图
        for i, (metric, values) in enumerate(metrics_data.items()):
            ax = axes[i]

            # 创建标识异常值的掩码
            is_outlier = []
            if metric == 'R²':
                # 对于R²，标记明显偏离[0,1]范围的值为异常值
                mean_value = np.mean([v for v in values if 0 <= v <= 1])
                is_outlier = [abs(v - mean_value) > 0.5 or v < -0.2 or v > 1.2 for v in values]
            else:
                # 对于其他指标，使用IQR方法检测异常值
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                is_outlier = [v < q1 - 1.5 * iqr or v > q3 + 1.5 * iqr for v in values]

            # 设置X轴标签和位置
            objective_names = [f'目标{j + 1}' for j in range(n_objectives)]
            x = np.arange(n_objectives)

            # 绘制条形图
            bars = ax.bar(x, values, width=0.6,
                          color=['#ff7f0e' if outlier else '#1f77b4' for outlier in is_outlier],
                          alpha=0.7)

            # 添加数值标签，避免重叠
            label_offsets = [0] * n_objectives  # 用于跟踪每个位置的标签偏移
            for j, (val, bar, outlier) in enumerate(zip(values, bars, is_outlier)):
                # 计算标签位置，异常值标签放在条形图内部
                if outlier and val > 0:
                    y_pos = val * 0.5  # 在条形图中间
                    color = 'white'  # 白色文本更易读
                    va = 'center'  # 垂直居中
                else:
                    y_pos = val + label_offsets[j]
                    color = 'black'  # 标准黑色文本
                    va = 'bottom'  # 在条形图上方
                    # 增加偏移量避免后续标签重叠
                    label_offsets[j] += max(abs(val) * 0.05, 0.02)

                # 添加标签，使用科学记数法显示极端值
                if abs(val) >= 1000 or abs(val) < 0.01:
                    text = f'{val:.2e}'
                else:
                    text = f'{val:.4f}'.rstrip('0').rstrip('.') if '.' in f'{val:.4f}' else f'{val:.0f}'

                ax.text(j, y_pos, text, ha='center', va=va, fontsize=9,
                        color=color, fontweight='bold' if outlier else 'normal')

            # 设置y轴范围，确保异常值也能显示，但不破坏图表的可读性
            normal_values = [v for v, o in zip(values, is_outlier) if not o]
            if normal_values:  # 如果有正常值
                y_min = min(0, min(normal_values) * 1.2)
                y_max = max(normal_values) * 1.2

                # 确保有足够空间显示标签
                y_max += max(label_offsets) * 2

                # 如果有异常值，在图表顶部显示它们的值
                if any(is_outlier):
                    for j, (val, outlier) in enumerate(zip(values, is_outlier)):
                        if outlier and (val < y_min or val > y_max):
                            # 在顶部添加注释，指出异常值
                            ax.annotate(f'异常值: {val:.4f}',
                                        xy=(j, y_max * 0.95),
                                        xytext=(j, y_max * 0.85),
                                        arrowprops=dict(arrowstyle='->'),
                                        ha='center', fontsize=8, color='red')

                ax.set_ylim(y_min, y_max)

            # 添加明确的坐标轴标签
            ax.set_xlabel('目标函数', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric}验证结果', fontsize=14)

            # 设置x轴刻度和标签
            ax.set_xticks(x)
            ax.set_xticklabels(objective_names, rotation=0)

            # 添加网格线，仅显示水平线以帮助读取数值
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # 为每个条形添加边框，使其更易区分
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(0.5)

        # 设置总标题
        fig.suptitle(f'代理模型验证指标 - 迭代 {iteration}', fontsize=16, fontweight='bold', y=0.98)

        # 调整子图间距，确保没有重叠
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, hspace=0.4)

        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"验证指标图表已保存至: {save_path}")

        # 返回图表对象供后续可能的修改
        return fig, axes


    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """计算各种验证指标"""
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        max_error = np.max(np.abs(y_true - y_pred))
        relative_error = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # 相对误差百分比

        return {
            'r2': r2,
            'rmse': rmse,
            'max_error': max_error,
            'relative_error': relative_error
        }

    def update_history(self, iteration, objectives, true_values, predicted_values):
        """
        更新验证历史记录

        参数:
        ----
        iteration : int
            当前迭代次数
        objectives : list
            目标函数名称列表 ['Cl', 'Cd', 'Cm']
        true_values : numpy.ndarray
            实际CFD计算的结果值，形状为 (n_samples, n_objectives)
        predicted_values : numpy.ndarray
            代理模型预测的结果值，形状为 (n_samples, n_objectives)
        """
        try:
            # 确保历史记录字典已初始化
            if not hasattr(self, 'history') or self.history is None:
                self.history = {}

            # 如果当前迭代不在历史记录中，则初始化
            if iteration not in self.history:
                self.history[iteration] = {}

            # 计算各个指标并保存
            for i, obj_name in enumerate(objectives):
                # 确保对象名的历史记录字典已初始化
                if obj_name not in self.history[iteration]:
                    self.history[iteration][obj_name] = {}

                # 获取当前目标函数的实际值和预测值
                y_true = true_values[:, i]
                y_pred = predicted_values[:, i]

                # 计算各种指标
                r2 = r2_score(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                max_err = max_error(y_true, y_pred)

                # 计算相对误差 (%)
                rel_error = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-10, None))) * 100

                # 保存指标
                self.history[iteration][obj_name] = {
                    'r2': r2,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'max_error': max_err,
                    'rel_error': rel_error
                }

            print(f"迭代 {iteration} 的验证历史记录已更新")

        except Exception as e:
            print(f"更新验证历史记录时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_validation_report(self, iteration=None):
        """将验证历史记录导出为CSV报表

        参数:
        ----
        iteration : int, 可选
            指定迭代次数，如果为None则导出所有迭代的报告
        """
        try:
            if not self.history:
                print("验证历史为空，无法导出报表")
                return

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 确定文件名
            if iteration is not None:
                filename = f'validation_report_iter{iteration}_{timestamp}.csv'
            else:
                filename = f'validation_report_{timestamp}.csv'

            save_path = os.path.join(self.save_dir, filename)

            # 创建报表数据
            rows = []
            headers = ['迭代次数', '目标函数', 'R²', 'MSE', 'RMSE', 'MAE', '最大误差', '相对误差(%)']

            # 确定要处理的迭代次数
            iterations = [iteration] if iteration is not None else sorted(self.history.keys())

            for iter_num in iterations:
                if iter_num not in self.history:
                    continue

                for obj_name in sorted(self.history[iter_num].keys()):
                    metrics = self.history[iter_num][obj_name]
                    row = [
                        iter_num,
                        obj_name,
                        metrics.get('r2', 'N/A'),
                        metrics.get('mse', 'N/A'),
                        metrics.get('rmse', 'N/A'),
                        metrics.get('mae', 'N/A'),
                        metrics.get('max_error', 'N/A'),
                        metrics.get('rel_error', 'N/A')
                    ]
                    rows.append(row)

            # 写入CSV文件
            import csv
            with open(save_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)

            print(f"验证报表已导出至: {save_path}")

            return save_path

        except Exception as e:
            print(f"导出验证报表时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None



    def save_validation_history(self, filename=None):
        """保存验证历史记录到文件

        参数:
        ----
        filename : str, 可选
            保存的文件名，如果为None则自动生成
        """
        try:
            if not self.history:
                print("验证历史为空，无需保存")
                return

            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'validation_history_{timestamp}.pkl'

            save_path = os.path.join(self.save_dir, filename)

            # 使用pickle保存历史数据
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(self.history, f)

            print(f"验证历史记录已保存至: {save_path}")

            # 同时保存为CSV格式以便查看
            self.export_validation_report()

        except Exception as e:
            print(f"保存验证历史记录时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def load_validation_history(self, filepath):
        """从文件加载验证历史记录

        参数:
        ----
        filepath : str
            历史记录文件路径

        返回:
        ----
        bool : 加载是否成功
        """
        try:
            if not os.path.exists(filepath):
                print(f"文件不存在: {filepath}")
                return False

            # 使用pickle加载历史数据
            import pickle
            with open(filepath, 'rb') as f:
                self.history = pickle.load(f)

            print(f"验证历史记录已从 {filepath} 加载")
            return True

        except Exception as e:
            print(f"加载验证历史记录时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def export_validation_report(self, filename=None):
        """将验证历史记录导出为CSV报表

        参数:
        ----
        filename : str, 可选
            保存的文件名，如果为None则自动生成
        """
        try:
            if not self.history:
                print("验证历史为空，无法导出报表")
                return

            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'validation_report_{timestamp}.csv'

            save_path = os.path.join(self.save_dir, filename)

            # 创建报表数据
            rows = []
            headers = ['迭代次数', '目标函数', 'R²', 'MSE', 'RMSE', 'MAE', '最大误差', '相对误差(%)']

            for iter_num in sorted(self.history.keys()):
                for obj_name in sorted(self.history[iter_num].keys()):
                    metrics = self.history[iter_num][obj_name]
                    row = [
                        iter_num,
                        obj_name,
                        metrics.get('r2', 'N/A'),
                        metrics.get('mse', 'N/A'),
                        metrics.get('rmse', 'N/A'),
                        metrics.get('mae', 'N/A'),
                        metrics.get('max_error', 'N/A'),
                        metrics.get('rel_error', 'N/A')
                    ]
                    rows.append(row)

            # 写入CSV文件
            import csv
            with open(save_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)

            print(f"验证报表已导出至: {save_path}")

        except Exception as e:
            print(f"导出验证报表时出错: {str(e)}")
            traceback.print_exc()

    def plot_model_comparison(self, save_dir=None):
        """绘制不同目标函数在各指标上的比较图

        参数:
        ----
        save_dir : str, 可选
            保存图表的目录，如果为None则使用self.save_dir
        """
        try:
            # 使用传入的save_dir或默认目录
            save_dir = save_dir or self.save_dir
            os.makedirs(save_dir, exist_ok=True)

            # 检查历史数据
            if not self.history:
                print("验证历史为空，无法绘图")
                return

            # 获取最新迭代次数
            latest_iter = max(self.history.keys())

            # 设置matplotlib参数
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False

            # 比较指标
            metrics = ['r2', 'rmse', 'rel_error', 'max_error']
            titles = ['R² 分数', 'RMSE', '相对误差 (%)', '最大误差']
            colors = {'Cl': 'blue', 'Cd': 'red', 'Cm': 'green'}

            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()

            # 获取所有目标函数
            obj_names = set()
            for iter_data in self.history.values():
                obj_names.update(iter_data.keys())

            obj_names = sorted(list(obj_names))

            for i, metric in enumerate(metrics):
                ax = axes[i]

                # 提取数据
                values = []
                labels = []
                colors_list = []

                for obj_name in obj_names:
                    for iter_num in sorted(self.history.keys()):
                        if obj_name in self.history[iter_num]:
                            if metric in self.history[iter_num][obj_name]:
                                values.append(self.history[iter_num][obj_name][metric])
                                labels.append(f"{obj_name} (iter {iter_num})")
                                colors_list.append(colors[obj_name])

                # 绘制条形图
                if values:
                    bars = ax.bar(range(len(values)), values, color=colors_list)

                    # 添加数值标签
                    for j, val in enumerate(values):
                        # 限制显示的小数位数，根据数值范围
                        if abs(val) > 10:
                            val_text = f"{val:.1f}"
                        elif abs(val) > 1:
                            val_text = f"{val:.2f}"
                        else:
                            val_text = f"{val:.3f}"

                        ax.text(j, val * 1.05, val_text,
                                ha='center', va='bottom',
                                fontsize=9, rotation=0)

                    # 设置x轴标签
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha='right')

                    # 设置标题和标签
                    ax.set_title(titles[i], fontsize=14)
                    ax.set_ylabel(metric, fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

                    # 为R²设置适当的y轴范围
                    if metric == 'r2':
                        ax.set_ylim([-0.1, 1.1])

            # 调整布局
            plt.tight_layout()

            # 保存图片
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'model_comparison_{timestamp}.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"模型比较图已保存至: {save_path}")

        except Exception as e:
            print(f"绘制模型比较图时出错: {str(e)}")
            traceback.print_exc()

    def plot_prediction_scatter(self, model, X_test, y_test, obj_name, iteration=None, save_dir=None):
        """绘制预测值与真实值的散点图

        参数:
        ----
        model : object
            训练好的代理模型
        X_test : np.ndarray
            测试数据的特征
        y_test : np.ndarray
            测试数据的目标值
        obj_name : str
            目标函数名称
        iteration : int, 可选
            模型的迭代次数
        save_dir : str, 可选
            保存图表的目录
        """
        try:
            # 使用传入的save_dir或默认目录
            save_dir = save_dir or self.save_dir
            os.makedirs(save_dir, exist_ok=True)

            # 使用模型进行预测
            y_pred = model.predict(X_test)

            # 计算相关指标
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # 设置matplotlib参数
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False

            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 8))

            # 绘制散点图
            ax.scatter(y_test, y_pred, alpha=0.7, s=50, edgecolor='k', c='blue')

            # 添加对角线
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.plot(lims, lims, 'k--', alpha=0.7, zorder=0)

            # 设置标题和标签
            iter_str = f" (迭代 {iteration})" if iteration is not None else ""
            ax.set_title(f'{obj_name} 代理模型预测值与真实值比较{iter_str}', fontsize=16)
            ax.set_xlabel('真实值', fontsize=14)
            ax.set_ylabel('预测值', fontsize=14)

            # 添加性能指标文本
            textstr = f'$R^2 = {r2:.3f}$\n$RMSE = {rmse:.3f}$'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

            # 调整布局
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            # 保存图片
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{obj_name}_prediction_scatter'
            if iteration is not None:
                filename += f'_iter{iteration}'
            filename += f'_{timestamp}.png'

            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"{obj_name} 预测散点图已保存至: {save_path}")

        except Exception as e:
            print(f"绘制预测散点图时出错: {str(e)}")
            traceback.print_exc()