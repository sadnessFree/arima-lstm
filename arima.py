import math
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import adfuller


def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def mean_a_p_e(y_true, y_pred):
    sum = 0
    for i in range(len(y_true)):
        sum += math.fabs(y_pred[i] - y_true[i]) / y_true[i]
    return sum / len(y_true)


series = read_csv('/Users/daihanru/Desktop/研究生小论文/时间序列数据9-8.csv', header=0, parse_dates=[0],
                  index_col=0,
                  squeeze=True,
                  date_parser=parser)
X = series.values
X = X.astype('float32')

size = 6000
train, test = X[0:size], X[size:size + 100]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(4, 0, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f RMSE:%.3f' % (error, math.sqrt(error)))
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

# def arima_predict(trainStart=0, trainEnd=5260, testStart=5260, testEnd=5560, step=1):
#     train, test = X[trainStart:trainEnd], X[testStart - step + 1:testEnd + step - 1]
#     history = [x for x in train]
#     predictions = list()
#     true = list()
#     for t in range(len(test) - step + 1):
#         model = ARIMA(history, order=(4, 0, 0))
#         model_fit = model.fit(disp=0)
#         output = model_fit.forecast(steps=step)
#         yhat = output[0]
#         predictions.append(yhat[step - 1])
#         obs = test[t + step - 1]
#         history.append(obs)
#         true.append(obs)
#         print('predicted=', yhat, 'expected=', obs)
#     mse = mean_squared_error(true, predictions)
#     mae = mean_absolute_error(true, predictions)
#     mape = mean_a_p_e(true, predictions)
#     print('Test MSE: %.3f' % mse)
#     print('Test RMSE: %.3f' % np.sqrt(mse))
#     print('Test MAE: %.3f' % mae)
#     print('Test MAPE: %.3f' % mape)
#     plt.plot(true, '-', label="TEST")
#     plt.plot(predictions, '--', color='red', label="ARIMA")
#     plt.axis([0, 300, 0, 30])
#     plt.legend(loc='upper left')
#     plt.show()
#     return true, predictions, np.sqrt(mse)
