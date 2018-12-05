import math
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import datetime
import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def mean_a_p_e(y_true, y_pred):
    sum = 0
    for i in range(len(y_true)):
        sum += math.fabs(y_pred[i] - y_true[i]) / y_true[i]
    return sum / len(y_true)


series = read_csv('/Users/daihanru/Desktop/arima-lstm/DataSet/FEB15-2.csv', usecols=[2, 3], header=0, parse_dates=[0],
                  index_col=0,
                  squeeze=True,
                  date_parser=parser)
X = series.values
X = X.astype('float32')

# x = test_stationarity(X)
# v, p, q, i = proper_model(X, 10)
start = 2000
size = 1100
test_size = 500
line_test_size = 300
windows_size = 10
test_start = line_test_size + windows_size - 1

arima_train, arima_test = X[start:start + size], X[start + size:start + size + test_size]
history = [x for x in arima_train]
predictions = list()
# order = st.arma_order_select_ic(arima_train, max_ar=5, max_ma=5, ic=['aic'])
# order.bic_min_order
# model = ARIMA(history, order=(5, 1, 1)).fit()
# predictions = model.predict(1000, 1019, dynamic=True)
for t in range(len(arima_test)):
    model = ARIMA(history, order=(4, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = arima_test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(arima_test, predictions)
mae = mean_absolute_error(arima_test, predictions)
mape = mean_a_p_e(arima_test, predictions)
print('ARIMA Test MAE:%.3f MSE: %.3f RMSE:%.3f MAPE:%.3f' % (mae, error, math.sqrt(error), mape))
# plot
plt.plot(arima_test[test_start:], '-', label="real flow")
plt.plot(predictions[test_start:], '--', color='red', label="ARIMA")
plt.legend(loc='upper right')
plt.xlabel("period(15-minute intervals)")
plt.ylabel("volume(vehicle/period)")
plt.ylim(0, 800)
plt.show(figsize=(12, 6))
