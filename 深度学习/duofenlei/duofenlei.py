"""
新闻分类：多分类问题
项目目标：把每个新闻分类到46个主题里，每个新闻只划分到一个主题：单标签多分类（若每个新闻能划分到多个主题：多标签多分类）
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras import models
from keras import layers

# 导入数据
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


# 好奇新闻内容解码为单词看看
# word_index = reuters.get_word_index()
# reverse_word_index = dict([value, key] for (key, value) in word_index.items())
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# 准备数据
def vectorize_sequences(sequences, demension=10000):
    results = np.zeros((len(sequences), demension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


def to_one_hot(labels, demension=46):
    results = np.zeros((len(labels), demension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)
# from keras.utils.np_utils import to_categorical
# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)

# 构建网络
# 第三层输出一个46维向量，使用了softmax激活（每篇新闻输出 在46个不同分类上 的概率分布，46个概率总和为1）
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# 选择优化器、损失函数和指标
# 损失函数：categorical_crossentropy （分类交叉熵），衡量网络输出的概率分布和标签的真实分布
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 留出验证集
x_val = x_train[:1000]
parital_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
parital_y_train = one_hot_train_labels[1000:]

# 训练模型
history = model.fit(parital_x_train,
                    parital_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 绘制看结果
history_dict = history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

results = model.evaluate(x_test, one_hot_test_labels)
print("训练后模型损失与精度：", results)

prediction = model.predict(x_test)
print("模型测试第一个新闻标签为：", np.argmax(prediction[0]))

