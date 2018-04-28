import pandas
import numpy
import math
import matplotlib.pyplot as plt
import arima
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# 加载数据
dataframe = pandas.read_csv('/Users/daihanru/Desktop/研究生小论文/时间序列数据9-8-2.csv', usecols=[1], engine='python',
                            skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

# 归一化数据
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

look_back = 10

test = dataset[arima.size - look_back:6121]


# 根据格式创建数据
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        # a = numpy.append(a, [dataset[i + look_back - 1, 1]])
        # a = numpy.append(a, [dataset[i + look_back - 1, 2] / 2])
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


testX, testY = create_dataset(test, look_back)
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

lstm_model = load_model('../model/lstm.h5')

testPredict = lstm_model.predict(testX)

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# testScore = lstm_model.evaluate(testX, testY, verbose=0)
testScore = mean_squared_error(arima.arima_test, testPredict)
# print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

combinePredict = []
combinePredict = numpy.append(combinePredict, ((arima.predictions + testPredict) / 2))
mean_squared_error(arima.arima_test, combinePredict)
print('combine Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

plt.plot(arima.arima_test, '-', label="real flow")
plt.plot(arima.predictions, '--', color='y', label="ARIMA")
plt.plot(testPredict, '--', color='red', label="LSTM")
plt.plot(combinePredict, 'x-', color='g', label="combined")
plt.legend(loc='upper left')
plt.show()
