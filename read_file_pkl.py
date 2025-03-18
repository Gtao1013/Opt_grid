import pickle
import numpy as np


# 读取模型文件函数
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# 模型文件路径
model_path = r"F:\essay_gt\opt\surrogate_models\Cm_model_iter_0_20250317_164613.pkl"

# 读取模型
try:
    model = load_model(model_path)
    print(f"模型类型: {type(model).__name__}")

    # 打印模型属性
    print("\n模型属性:")
    for attr in dir(model):
        if not attr.startswith('_') and attr not in ['predict', 'fit', 'score']:
            try:
                value = getattr(model, attr)
                if not callable(value):
                    print(f"{attr}: {value}")
            except:
                pass

    # 如果要使用模型进行预测，可以这样做:
    # 创建一个测试点（需要符合模型输入维度）
    test_point = np.array([[400, 1200, 25]])  # 假设模型输入是三维的: [Chord, Distance, Fai]

    # 进行预测
    prediction = model.predict(test_point)
    print(f"\n测试点 {test_point[0]} 的预测结果: {prediction[0]}")

except Exception as e:
    print(f"读取模型时出错: {str(e)}")