# from sklearn.feature_extraction import DictVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# import jieba
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.decomposition import PCA
# 
# def pca():
#     """
#     主成分分析
#     :return:None
#     """
#     pca = PCA(n_components=9)
#
#     data = pca.fit_transform([[0,1,2], [0,2,3], [0,3,4]])
#
#     print(data)
#
#     return None
#
#
# def var():
#     """
#     特征选择 - 过滤式
#     :return: None
#     """
#     var = VarianceThreshold(threshold=0.0)
#
#     data = var.fit_transform([[0,1,2], [0,2,3], [0,3,4]])
#
#     print(data)
#
#     return None
#
#
# def stand():
#     """
#     数据预处理 - 标准化
#     :return: None
#     """
#     std = StandardScaler()
#
#     data = std.fit_transform([[1, -2, 3], [1, 2, 3]])
#
#     print(data)
#
#     return None
#
#
# def mm():
#     """
#     数据预处理 - 归一化
#     :return:None
#     """
#
#     mm = MinMaxScaler(feature_range=(2, 3))
#
#     data = mm.fit_transform([[1, -2, 3], [1, 2, 3]])
#
#     print(data)
#
#     return None
#
#
# def tfidfvec():
#     c = cutword()
#     print("jieba分词结果：", c)
#
#     tf = TfidfVectorizer()
#     data = tf.fit_transform(["life is is", "today is so so"])
#
#     print(tf.get_feature_names())
#     print(data.toarray())
#
#     return None
#
#
# def countvec():
#     """
#     文本特征抽取
#     :return: None
#     """
#     c = cutword()
#     print("jieba分词结果：", c)
#
#     cv = CountVectorizer()
#     data = cv.fit_transform(["life is is short，i like python", "today is so so", c])
#
#     print(cv.get_feature_names())
#     print(data.toarray())
#
#     return None
#
#
# def cutword():
#     con = jieba.cut("人生苦短，我需要简单的社会需要的语言")
#
#     # 转换成列表
#     content = list(con)
#
#     # 列表转换成字符串
#     c = ' '.join(content)
#
#     return c
#
#
# def dictvec():
#     """
#     字典数据抽取
#     :return: None
#     """
#     # 实例化
#     dict = DictVectorizer(sparse=False)
#
#     # 调用fit_transform(X)
#     data = dict.fit_transform([{"name": "hyl", "age": 23}, {"name": "lyh", "age": 22}])
#
#     # 调用get_feature_names()查看类别名称
#     print(dict.get_feature_names())
#     print(data)
#
#     return None
#
#
# if __name__ == '__main__':
#     pca()
