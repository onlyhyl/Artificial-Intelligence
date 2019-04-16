"""
电影评论分类：二分类问题
项目目标：根据电影评论的文字内容划分为正面或负面
"""

from keras.datasets import imdb
from keras import models
from keras import layers
# from keras import optimizers
# from keras import losses
# from keras import metrics
import numpy as np
import matplotlib.pyplot as plt

# 导入数据
# 1）加载imdb(互联网电影数据库)数据集，包含50000条两极分化评论
#    num_words保留训练集10000个最常出现的词（labels中1：正面，2负面）
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# train_data = [list([1, 14, 22, 16, .., 178, 32])...list([1, 17, 6,...,1]))
# train_labels = [1 0 0 ... 0 1 0]

# 2）将单词映射为整数索引的字典
word_index = imdb.get_word_index()
# word_index = {'reverent': 44834, 'gangland': 22426, ..., "'ogre'": 65029}

# 3）键值颠倒，将整数索引映射为单词
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
# word_index.items() = ([('dayan', 84842), ("fortinbras'", 67418), ...,('characther', 67419), ("emotion's", 67420),])
# reverse_word_index = {44834: 'reverent', 22426: 'gangland', ..., 65029: "'ogre'"}

# 4）将评论解码
#   索引-3（0,1,2是为“padding”(填充)、“start of sequence”(序列开始)、“unknown”(未知词)分别保留的索引）
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# decoded_review = this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert ? ...?


# 准备数据
# 1）将整数序列编码编码为二进制矩阵( 不能将整数序列直接输入神经网络。你需要将列表转换为张量 )
def vectorize_sequences(sequences, dimension=10000):
    # 创建一个形状为(len(sequences), dimension)的零矩阵
    results = np.zeros((len(sequences), dimension))
    # 将results[i]的指定索引设为1
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# 2)将训练集和测试集数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# x_train、x_test = [[0. 0. 0. ... 0. 0. 0.][0. 1. 1. ... 0. 0. 0.]..[0. 1. 1. ... 0. 0. 0.]]

# 3)将目标值(电影)向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
# y_train、y_test =[[0. 0. 0. ... 0. 0. 0.][0. 1. 1. ... 0. 0. 0.]..[0. 1. 1. ... 0. 0. 0.]]



# 构建网络
# 网络有三层。两层中间层，每层16个隐层单元；第三层输出一个标量，预测当前评论情感
#   中间层使用relu 作为激活函数，最后一层则针对二分类问题使用sigmoid 激活以输出一个0~1 范围内的概率
#   relu函数将所有负值归零，而sigmoid 函数则将任意值“压缩”到[0,1] 区间内
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 选择优化器、损失函数和指标
# 损失函数选择：对于二分类最好选交叉熵（用于衡量概率分布之间的距离），如binary_crossentropy （二元交叉熵）损失，衡量真实分布与预测值之间的距离
# 指标：训练过程中监控指标
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 或
# 使用自定义优化器、损失和指标
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])

# 留出10000个样本作为验证集(x_val、y_val)
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 训练模型
#   batch_size使用512 个样本组成的小批量，epochs将模型训练20 个轮次（即对x_train 和y_train 两个张量中的所有样本进行20 次迭代）。
#   将验证数据传入validation_data 参数，监控验证集的损失和精度。
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# # 绘制看结果
# # 1）绘制训练损失和验证损失
# history_dict = history.history
# # history_dict.keys = dict_keys(['val_acc', 'acc', 'val_loss', 'loss']) (val_acc:验证精度，acc训练精度)
# loss_values = history_dict['loss']
# history_dict['loss'] = [0.5084604772408803, 0.3005697673479716, 0.2179936485528946, 0.17508196938037873, 0.14274606738885243, 0.1149827463944753, 0.09792886832952499...]
# val_loss_values = history_dict['val_loss']
#
# epochs = range(1, len(loss_values)+1)
# # epochs = range(1, 21)
#
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.show()
#
# # 2）绘制训练精度和验证精度
# # clf():清空图
# plt.clf()
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and Validation acc')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
## 由绘制图可看出每轮训练损失↓精度↑，但验证中并非如此，大概在第4轮达到最佳值：过拟合现象（因为在第2轮后对训练数据过度优化，最终学到的仅限于训练数据，无法泛化到其它数据）
## 需要用XXX方法来降低过拟合

# 用测试集查看模型训练结果
# results = model.evaluate(x_test, y_test)
# results = [0.8064336129784584, 0.84548](loss, acc)

# 训练好网络，用于实践
# model.predict(x_test)
# 用predict方法来得到评论为正面的可能性
# modef.predict(x_test) =[[7.9554915e-03]
#                         [9.9999988e-01]
#                         [9.5130897e-01]
#                         ...
#                         [9.9810958e-04]
#                         [3.6877692e-03]
#                         [6.0508829e-01]]


# 重新构建网络
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=4, batch_size=512)


history_dict = history.history
loss = history_dict['loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'b', label='Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history_dict['acc']
plt.plot(epochs, acc, 'b', label='Acc')
plt.title('Acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)
# results = [0.3247204120445251, 0.87288](loss, acc)

model.predict(x_test)
# model.predict(x_test)= [[0.14049816]
#                         [0.99970233]
#                         [0.29415238]
#                         ...
#                         [0.07267842]
#                         [0.04335585]
#                         [0.48002273]]
