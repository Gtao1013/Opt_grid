# model_utils.py
import pickle
import numpy as np
import os
import glob
from datetime import datetime


class SurrogateModelLoader:
    """用于加载和使用已保存代理模型的类"""

    def __init__(self, base_path=None):
        """初始化模型加载器

        参数:
        ----
        base_path : str, 可选
            模型文件的基础路径，默认为当前目录下的'surrogate_models'
        """
        if base_path is None:
            self.base_path = os.path.join(os.getcwd(), 'surrogate_models')
        else:
            self.base_path = os.path.join(base_path, 'surrogate_models')

        if not os.path.exists(self.base_path):
            raise ValueError(f"模型目录不存在: {self.base_path}")

    def get_available_models(self, obj_name=None):
        """获取可用的模型文件列表

        参数:
        ----
        obj_name : str, 可选
            目标函数名称，例如'Cl', 'Cd', 'Cm'

        返回:
        ----
        dict : 模型文件信息，按照迭代次数和时间戳组织
        """
        pattern = f"{obj_name}_model_*.pkl" if obj_name else "*_model_*.pkl"
        model_files = glob.glob(os.path.join(self.base_path, pattern))

        models_info = {}
        for file_path in model_files:
            file_name = os.path.basename(file_path)
            # 解析文件名以获取目标名称、迭代次数和时间戳
            parts = file_name.split('_')
            if len(parts) >= 4:
                obj = parts[0]
                iter_num = int(parts[2]) if parts[2].isdigit() else 0
                timestamp = '_'.join(parts[3:]).replace('.pkl', '')

                if obj not in models_info:
                    models_info[obj] = []
                models_info[obj].append({
                    'iteration': iter_num,
                    'timestamp': timestamp,
                    'file_path': file_path
                })

        # 按迭代次数排序
        for obj in models_info:
            models_info[obj].sort(key=lambda x: x['iteration'])

        return models_info

    def load_model(self, file_path):
        """加载模型文件

        参数:
        ----
        file_path : str
            模型文件的完整路径

        返回:
        ----
        model : 加载的模型对象
        """
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise

    def get_latest_model(self, obj_name):
        """获取指定目标函数的最新模型

        参数:
        ----
        obj_name : str
            目标函数名称，例如'Cl', 'Cd', 'Cm'

        返回:
        ----
        tuple : (model, iteration, timestamp) - 模型对象、迭代次数和时间戳
        """
        models_info = self.get_available_models(obj_name)

        if obj_name not in models_info or not models_info[obj_name]:
            raise ValueError(f"找不到{obj_name}的模型文件")

        # 获取最新的模型（最大迭代次数）
        latest_model_info = models_info[obj_name][-1]
        model = self.load_model(latest_model_info['file_path'])

        return model, latest_model_info['iteration'], latest_model_info['timestamp']

    def get_model_by_iteration(self, obj_name, iteration):
        """获取指定迭代次数的模型

        参数:
        ----
        obj_name : str
            目标函数名称，例如'Cl', 'Cd', 'Cm'
        iteration : int
            迭代次数

        返回:
        ----
        model : 加载的模型对象
        """
        models_info = self.get_available_models(obj_name)

        if obj_name not in models_info:
            raise ValueError(f"找不到{obj_name}的模型文件")

        # 查找指定迭代的模型
        for model_info in models_info[obj_name]:
            if model_info['iteration'] == iteration:
                return self.load_model(model_info['file_path'])

        raise ValueError(f"找不到{obj_name}的第{iteration}次迭代模型")

    def predict(self, model, x_new):
        """使用模型进行预测

        参数:
        ----
        model : 模型对象
        x_new : np.ndarray
            新的输入点，形状为 (n_samples, n_features)

        返回:
        ----
        np.ndarray : 预测结果
        """
        x_new = np.atleast_2d(x_new)
        return model.predict(x_new)

    def batch_predict(self, models_dict, x_new):
        """使用多个模型批量预测

        参数:
        ----
        models_dict : dict
            键为目标名称，值为模型对象的字典
        x_new : np.ndarray
            新的输入点，形状为 (n_samples, n_features)

        返回:
        ----
        dict : 键为目标名称，值为预测结果的字典
        """
        x_new = np.atleast_2d(x_new)
        results = {}

        for obj_name, model in models_dict.items():
            results[obj_name] = self.predict(model, x_new)

        return results

    def load_all_latest_models(self):
        """加载所有目标函数的最新模型

        返回:
        ----
        dict : 键为目标名称，值为模型对象的字典
        """
        models_info = self.get_available_models()
        models = {}

        for obj_name in models_info:
            model, _, _ = self.get_latest_model(obj_name)
            models[obj_name] = model

        return models


# 示例用法
def example_usage():
    # 初始化模型加载器
    base_path = r'F:\essay_gt\opt'
    loader = SurrogateModelLoader(base_path)

    try:
        # 1. 获取可用模型信息
        models_info = loader.get_available_models()
        print("可用模型信息:")
        for obj_name, models in models_info.items():
            print(f"\n{obj_name} 模型:")
            for model in models:
                print(f"  迭代: {model['iteration']}, 时间戳: {model['timestamp']}")

        # 2. 加载最新的Cl模型
        cl_model, iteration, timestamp = loader.get_latest_model('Cl')
        print(f"\n加载了Cl的最新模型 (迭代: {iteration}, 时间戳: {timestamp})")

        # 3. 使用模型进行预测
        test_point = np.array([[400, 1200, 25]])  # [Chord, Distance, Fai]
        prediction = loader.predict(cl_model, test_point)
        print(f"\n测试点 {test_point[0]} 的Cl预测: {prediction[0]}")

        # 4. 加载所有目标的最新模型
        all_models = loader.load_all_latest_models()
        print(f"\n已加载所有目标的最新模型: {list(all_models.keys())}")

        # 5. 批量预测
        batch_results = loader.batch_predict(all_models, test_point)
        print("\n批量预测结果:")
        for obj_name, pred in batch_results.items():
            print(f"  {obj_name}: {pred[0]}")

    except Exception as e:
        print(f"示例运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    example_usage()