import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler


class KrigingModel:
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.model = None
        # 使用StandardScaler进行标准化
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        # 仍保留这些属性以兼容现有代码
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def fit(self, X, y):
        """训练Kriging模型，增加数据标准化和更稳健的核函数"""
        X_train = np.copy(X)
        y_train = np.copy(y).reshape(-1)  # 确保y是一维数组

        # 使用StandardScaler进行标准化
        if self.normalize:
            # 使用scaler拟合并转换X
            self.scaler_X.fit(X_train)
            X_scaled = self.scaler_X.transform(X_train)

            # 保存原始数据的统计信息，用于兼容旧代码
            self.X_mean = np.mean(X_train, axis=0)
            self.X_std = np.std(X_train, axis=0)
            # 避免除以零
            self.X_std = np.where(self.X_std < 1e-10, 1.0, self.X_std)

            # 对于y，我们使用reshape确保它是2D数组
            y_reshaped = y_train.reshape(-1, 1)
            self.scaler_y.fit(y_reshaped)
            y_scaled = self.scaler_y.transform(y_reshaped).ravel()  # 转回1D数组

            # 保存y的统计信息，用于兼容旧代码
            self.y_mean = np.mean(y_train)
            self.y_std = np.std(y_train)
            self.y_std = max(self.y_std, 1e-10)
        else:
            X_scaled = X_train
            y_scaled = y_train

        # 创建更稳健的核函数
        # 使用Matern核函数 + 白噪声
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0] * X_scaled.shape[1],
                                              length_scale_bounds=(1e-3, 1e3),
                                              nu=1.5) + WhiteKernel(1e-5, (1e-10, 1e-1))

        # 初始化高斯过程回归器
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-10,
            normalize_y=False,  # 我们已经手动标准化了
            random_state=42
        )

        # 训练模型
        try:
            self.model.fit(X_scaled, y_scaled)
            print(f"核函数参数: {self.model.kernel_}")
            return self
        except Exception as e:
            print(f"模型训练失败: {str(e)}")
            raise

    def predict(self, X):
        """使用训练好的模型进行预测"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        X_test = np.copy(X)

        # 对输入进行与训练数据相同的标准化
        if self.normalize:
            X_scaled = self.scaler_X.transform(X_test)
            y_pred = self.model.predict(X_scaled)

            # 反标准化预测结果
            y_pred_reshaped = y_pred.reshape(-1, 1)
            y_pred = self.scaler_y.inverse_transform(y_pred_reshaped).ravel()
        else:
            y_pred = self.model.predict(X_test)

        return y_pred

    def predict_std(self, X):
        """预测给定点的标准差，用于探索采样"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        X_test = np.copy(X)

        # 对输入进行标准化
        if self.normalize:
            X_scaled = self.scaler_X.transform(X_test)
            y_pred, y_std = self.model.predict(X_scaled, return_std=True)

            # 反标准化标准差
            y_std = y_std * self.y_std
        else:
            _, y_std = self.model.predict(X_test, return_std=True)

        return y_std

    def score(self, X, y):
        """计算模型在给定数据上的R²分数"""
        try:
            y_pred = self.predict(X)
            return r2_score(y, y_pred)
        except Exception as e:
            print(f"计算得分时出错: {str(e)}")
            raise