import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# # 固定让模能复现
# numpy.random.seed(7)

# 加载数据
dataframe = pandas.read_csv('/Users/daihanru/Desktop/研究生小论文/时间序列数据9-8-2.csv', usecols=[1, 2, 3], engine='python',
                            skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

# 归一化数据
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

# 分为训练和测试数据
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))


# 根据格式创建数据
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        a = numpy.append(a, [dataset[i + look_back - 1, 1]])
        a = numpy.append(a, [dataset[i + look_back - 1, 2] / 2])
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# 根据look_back调整数据集 如look_back=3 则数据的一行为 t-3 t-2 t-1 t
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 创建多层的lstm模型
model = Sequential()
model.add(Dense(8, input_dim=look_back + 2, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=1000, batch_size=2, verbose=2)
model.save('../model/lstm.h5')

# 预测
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# 评价model效果
# trainScore = mean_squared_error(trainY[0], trainPredict[:, 0])
# testScore = mean_squared_error(testY[0], testPredict[:, 0])
testScore = model.evaluate(testX, testY, verbose=0)
# print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

# 画数据
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

plt.plot(dataset[6000:6100, 0], '-', label="real flow")
plt.plot(testPredictPlot[6000:6100, 0], '--', color='red', label="LSTM")
plt.legend(loc='upper left')
plt.show()
