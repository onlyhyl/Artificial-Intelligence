import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

def decision():
    """
    决策树预测泰坦尼克号上的生死
    :return: None
    """
    # 读取数据
    data = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    # 处理数据
    # 1)缺失值
    data['age'].fillna(data['age'].mean(), inplace=True)
    # 2)划分特征值与目标值
    x = data[['pclass', 'age', 'sex']]
    y = data['survived']

    # 分割成训练集与测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程 - 字典特征抽取
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    x_test = dict.transform(x_test.to_dict(orient="records"))

    # 使用算法
    dec = DecisionTreeClassifier(max_depth=8)
    dec.fit(x_train, y_train)
    y_predict = dec.predict(x_test)
    print("预测结果：", y_predict)
    print("预测的准确率：", dec.score(x_test, y_test))

    # 导出决策树
    export_graphviz(dec, out_file="./tree.dot", feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])

    return None
if __name__ == "__main__":
    decision()