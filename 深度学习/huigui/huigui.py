"""
房价预测：回归问题
项目目标：根据犯罪率、当地房产税等预测20世纪70年代中期波士顿郊区房屋价格的中位数。
"""


from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 只有 506 个数据点（分为 404 个训练样本和 102 个测试样本）输入数据的每个特征（比如犯罪率）都有不同的取值范围。
# 例如，有些特性是比例，取值范围为 0~1；有的取值范围为 1~12；还有的取值范围为 0~100
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
# train_targets = array([ 15.2, 42.3, 50. ... 19.4, 19.4, 29.1])（单位为千美元）

# 所以需要对特征值标准化（训练集）
# （即对于输入数据的每个特征，减去特征平均值，再除以标准差，这样得到的特征平均值为0，标准差为 1）
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# 损失函数：均方误差（MSE，mean squared error），预测值与目标值之差的平方（回归问题常用的损失函数）。
# 监控值：平均绝对误差（MAE，mean absolute error）。预测值与目标值之差的绝对值。
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=['mae'])
    return model

# 因数据点很少，验证集会非常小，导致验证分数可能会有很大波动
# 最佳做法是使用相应的验证方法：K 折交叉验证（将每个模型在K-1个分区上训练，并在剩下的一个分区上进行评估。）
# K=4：一共4折
# k = 4
# num_val_samples = len(train_data) // k
# num_epochs = 100
# all_scores = []
# for i in range(k):
#     print('processing fold #', i)
#     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
#
#     partial_train_data = np.concatenate(
#     [train_data[:i * num_val_samples],
#     train_data[(i + 1) * num_val_samples:]],
#     axis=0)
#
#     partial_train_targets = np.concatenate(
#     [train_targets[:i * num_val_samples],
#     train_targets[(i + 1) * num_val_samples:]],
#     axis=0)
#
# # verbose=0：静默模式
#     model = build_model()
#     model.fit(partial_train_data,
#               partial_train_targets,
#               epochs=num_epochs,
#               batch_size=1,
#               verbose=0)
#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
#     all_scores.append(val_mae)
#     np.mean(all_scores)
## all_scores = [2.0796982427634814, 2.2088884315868413, 2.9905093778478036, 2.3807723174000732]
## all_scores.mean = 2.9947904173572462
# 预测与实际相差很大（3000美元），所以尝试让训练时间更长一些

# # 修改训练循环
# num_epochs = 500
# all_mae_histories = []
# for i in range(k):
#     print('processing fold #', i)
#     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
#
#     partial_train_data = np.concatenate(
#     [train_data[:i * num_val_samples],
#     train_data[(i + 1) * num_val_samples:]],
#     axis=0)
#
#     partial_train_targets = np.concatenate(
#     [train_targets[:i * num_val_samples],
#     train_targets[(i + 1) * num_val_samples:]],
#     axis=0)
#
#     model = build_model()
#     history = model.fit(
#         partial_train_data,
#         partial_train_targets,
#         validation_data=(val_data, val_targets),
#         epochs=num_epochs,
#         batch_size=1,
#         verbose=0)
# # history.history.keys() = dict_keys(['val_loss', 'val_mean_absolute_error', 'loss', 'mean_absolute_error'])（若无验证集，则dict_keys(['loss', 'mean_absolute_error'])）
#     mae_history = history.history['val_mean_absolute_error']
#     all_mae_histories.append(mae_history)
#     print(all_mae_histories)
#
# # 计算所有轮次中的 K 折验证分数平均值
# average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# print(average_mae_history)
#
# plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()
#
# # 因为纵轴的范围较大，且数据方差相对较大，难以看清这张图的规律。我们来重新绘制一张图。
# # 1)删除前 10 个数据点，因为它们的取值范围与曲线上的其他点不同。
# # 2)将每个数据点替换为前面数据点的指数移动平均值，以得到光滑的曲线。
# def smooth_curve(points, factor=0.9):
#     smoothed_points = []
#     for point in points:
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(previous * factor + point * (1 - factor))
#         else:
#             smoothed_points.append(point)
#     return smoothed_points
# smooth_mae_history = smooth_curve(average_mae_history[5:])
#
# plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()
# # 验证 MAE 在 80 轮后不再显著降低，之后就开始过拟合。
# # 到此就完成了模型调参（轮数、隐藏层大小），使用最佳参数在所有训练数据上训练出最终模型

model = build_model()
model.fit(train_data,
          train_targets,
          epochs=80,
          batch_size=16,
          verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)

