import math
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def mean_a_p_e(y_true, y_pred):
    sum = 0
    for i in range(len(y_true)):
        sum += math.fabs(y_pred[i] - y_true[i]) / y_true[i]
    return sum / len(y_true)


series = read_csv('/Users/daihanru/Desktop/arima-lstm/DataSet/FEB15.csv', usecols=[2, 3], header=0, parse_dates=[0],
                  index_col=0,
                  squeeze=True,
                  date_parser=parser)
X = series.values
X = X.astype('float32')

# x = test_stationarity(X)
# v, p, q, i = proper_model(X, 10)
size = 1310
test_size = 10
arima_train, arima_test = X[0:size], X[1310:size + test_size + 1]
history = [x for x in arima_train]
predictions = list()
# model = ARIMA(history, order=(5, 1, 1)).fit()
# predictions = model.predict(1000, 1019, dynamic=True)
for t in range(len(arima_test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = arima_test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(arima_test, predictions)
print('ARIMA Test MSE: %.3f RMSE:%.3f' % (error, math.sqrt(error)))
# plot
plt.plot(arima_test, '-', label="real flow")
plt.plot(predictions, '--', color='red', label="ARIMA")
plt.legend(loc='upper left')
plt.show(figsize=(12, 6))
