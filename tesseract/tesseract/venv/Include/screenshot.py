import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%m-%Y')
import statsmodels.tsa.arima.model as stats
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import pickle
data = pd.read_csv('maindata.csv', sep=',', index_col='Date', parse_dates=['Date'], date_parser=dateparse).fillna(0)
# plt.figure(figsize=(10,6))
# df_close = data['Close']
# df_close.plot(style='k.',kind='hist')
# plt.title('Hisogram of closing price')
# plt.show()

#print(data.head())
df1 = data.iloc[:605,:]
df2 = data.iloc[606:,:]
#print("Shape of new dataframes - {} , {}".format(df1.shape, df2.shape))

#print(df1.tail())
#print(df2.head())

# plt.figure(figsize=(10,6))
# plt.grid(True)
# plt.xlabel('Dates')
# plt.ylabel('Close Prices')
# plt.plot(df1['Close'])
# plt.title('NasDaq closing price')
# plt.show()
#
# plt.figure(figsize=(10,6))
# plt.grid(True)
# plt.xlabel('Dates')
# plt.ylabel('Volume Traded')
# plt.plot(df1['Volume'])
# plt.title('NasDaq Volume Traded')
# plt.show()
#
# plt.figure(figsize=(10,6))
# plt.grid(True)
# plt.xlabel('Dates')
# plt.ylabel('Close Prices')
# plt.plot(df2['Close'])
# plt.title('NasDaq closing price')
# plt.show()
#
# plt.figure(figsize=(10,6))
# plt.grid(True)
# plt.xlabel('Dates')
# plt.ylabel('Volume Traded')
# plt.plot(df2['Volume'])
# plt.title('NasDaq Volume Traded')
# plt.show()

main1 = data['Close']
print(main1)
main2 = data['Volume']
print(main2)
# ADF Test
result = adfuller(main1, autolag='AIC')

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
if result[0] < result[4]["5%"]:
    print ("Reject Ho - Time Series is Stationary")
else:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")

result = adfuller(main2, autolag='AIC')

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
if result[0] < result[4]["5%"]:
    print ("Reject Ho - Time Series is Stationary")
else:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")

result = sm.tsa.seasonal_decompose(main1, model='multiplicative', period = 30)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(16, 9)

from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
df_log = np.log(main1)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average')
plt.plot(std_dev, color ="black", label = "Standard Deviation")
plt.plot(moving_avg, color="red", label = "Mean")
plt.legend()
plt.show()

train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
print(train_data)
print(test_data)
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_log, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()

model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
print(model_autoARIMA.summary())
model_autoARIMA.plot_diagnostics(figsize=(15,8))
plt.show()

model = stats.ARIMA(train_data, order=(2, 1, 0))
fitted = model.fit()
print(fitted.summary())

filename = 'ARIMA_Model.sav'
pickle.dump(fitted, open(filename, 'wb')) ## This will create a pickle file

model1 = pickle.load(open(filename, 'rb'))

print(model1)

start_index = len(train_data)
end_index = len(train_data) + len(test_data) - 1

# Predictions for one-year against the test set
forecast = model1.predict(start=start_index, end=end_index)
print(forecast)
plt.plot(train_data, label='training')
plt.plot(test_data, color = 'blue', label='Actual Stock Price')
#plt.plot(forecast, color = 'orange',label='Predicted Stock Price')
plt.title('S&P Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()