from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

def linear():
    """
    线性回归直接预测房子价格
    :return:None
    """
    # 导入数据（数据集已分好特征值与目标值）
    lb = load_boston()

    # 分割训练集与测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    # 特征工程 - 标准化
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # 算法运用
    # 正规方程求w权重预测结果
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print("正规方程的权重：", lr.coef_)
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
    print("正规方程预测房子的价格：", y_lr_predict)
    # 误差评估
    lrerror = mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict)
    print("正规方程的均方误差：", lrerror)


    # 梯度下降预测结果
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print("梯度下降的权重：", sgd.coef_)
    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    print("梯度下降预测房子的价格：", y_sgd_predict)
    # 误差评估
    sgderror = mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict)
    print("梯度下降的均方误差：", sgderror)

    return None

if __name__ == '__main__':
    linear()