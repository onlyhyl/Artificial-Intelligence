from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def knncls():
    """
    knn预测用户签到位置
    :return:None
    """
    # 读取数据
    data = pd.read_csv("./facebook-v-predicting-check-ins/train.csv")

    # 处理数据
    # 1)缩小数据量（注：为了学习效率，由于本主机配置原因作此操作，正常业务无需此操作）
    data = data.query("x > 1.0 & x < 1.1 & y > 2.5 & y < 2.6")

    # 2)处理时间戳（time）
    time_value = pd.to_datetime(data['time'], unit='s')
    # 2)转换成能成为特征值的数据形式，即字典格式
    time_value = pd.DatetimeIndex(time_value)
    # 2)用处理后的时间数据构建新特征
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday
    # 2)删除时间戳(axis=1以列删除)
    data = data.drop(['time'], axis=1)

    # 3)把签到数量少于n个目标位置（注：为了学习另外知识点，如筛选目标位置进行推荐做此操作）
    # groupby()按照该特征进行排序， count()统计其它特征的次数，
    # 即第一列为索引列(place_id，代码无法获取索引列)，其它列为每个特征值出现次数
    place_count = data.groupby('place_id').count()
    # reset_index()重设，即把索引列(place_id)重新作为回特征
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]
    data = data.drop(['row_id'], axis=1)

    # 3)取出特征值与目标值
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)

    # 4)分割成训练集与测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程（标准化）
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 运用算法
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)
    print("预测的签到位置：", y_predict)
    print(y_test)
    # print("预测的准确率：", knn.score(x_test, y_test))

    # 对knn参数进行交叉验证
    # param = {"n_neighbors": [3, 5, 7]}
    # gc = GridSearchCV(knn, param_grid=param, cv=10)
    # gc.fit(x_train, y_train)
    # print("交叉验证的准确率：", gc.score(x_test, y_test))
    # print("交叉验证的最好结果：", gc.best_score_)
    # print("交叉验证中最好的模型", gc.best_estimator_)
    # print("每次交叉验证的结果", gc.cv_results_)

    return None
if __name__ == "__main__":
    knncls()
