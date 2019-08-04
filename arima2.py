import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'red'


df = pd.read_excel("Superstore.xls")
technology = df.loc[df['Category'] == 'Technology']


print("start date:-{} , end date:-{} ".format(technology['Order Date'].min(),technology['Order Date'].max()))


cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
technology.drop(cols, axis=1, inplace=True)
technology = technology.sort_values('Order Date')

print("\n\nChecking for missing values:-")
print(technology.isnull().sum())

technology = technology.groupby('Order Date')['Sales'].sum().reset_index()

print("\n\n   Technology dataframe head")
print(technology.head())
print("\n\n   Technology dataframe tail")
print(technology.tail())

technology = technology.set_index('Order Date')
print("\n\n   technology index(new)")
print(technology.index)

#Actual Data
print("\n\n Actual data(Irregular Time Series)")
technology.plot(figsize=(15, 6))
plt.show()


#Month Start (Converting to regular time series)
y = technology['Sales'].resample('MS').mean()
print("\n\nMonth Start of 2017(Regular Time Series)")
print(y['2017':])

#Visualizing Techology Sales Time Series Data
print("\n\n Visualizing Office Supplies Sales")
y.plot(figsize=(15, 6))
plt.show()


from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

#decompsition of time series data to its components
print("\n\n Decomposition to its Components")
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

# Combination of parameters
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for ARIMA Model...')
print('(p,d,q),(P,D,Q): {} , {}'.format(pdq[1], seasonal_pdq[1]))
print('(p,d,q),(P,D,Q): {} , {}'.format(pdq[1], seasonal_pdq[2]))
print('(p,d,q),(P,D,Q): {} , {}'.format(pdq[5], seasonal_pdq[3]))
print('(p,d,q),(P,D,Q): {} , {}'.format(pdq[6], seasonal_pdq[4]))

#parameter combination selection
print("\n\nSelection of parameter combination based on Information Criteria")
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{},{} - BIC:{}'.format(param, param_seasonal, results.bic))
        except:
            continue

print("\nBest model so far ARIMA(1,1,1)(1,1,0,12)")
# Fitting the model
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print("\n\nSummary of the Selected Model")
print(results.summary())

#Residual of the model
results.plot_diagnostics(figsize=(16, 8))
plt.show()

# Validating forecasts
print("\n\n Validating forecast")
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()

ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Technology Sales')
plt.legend()

plt.show()

#The Root Mean Squared Error of our forecasts
y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


#Producing and visualizing forecasts
print("\nProducing and visualizing forecasts")
pred_uc = results.get_forecast(steps=25)
pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Technology Sales')

plt.legend()
plt.show()