# Sales Pridiction using ARIMA Model

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
matplotlib.rcParams['text.color'] = 'k'


df = pd.read_excel("Superstore.xls")
furniture = df.loc[df['Category'] == 'Furniture']


print("start date:-{} , end date:-{} ".format(furniture['Order Date'].min(),furniture['Order Date'].max()))


cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')

# Checking for Missing values
print("\n\nChecking for missing values:-")
print(furniture.isnull().sum())

furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

print("\n\n   Furniture dataframe head")
print(furniture.head())
print("\n\n   Furniture dataframe tail")
print(furniture.tail())

furniture = furniture.set_index('Order Date')
print("\n\n   Furniture index(new)")
print(furniture.index)

#Month Start (Converting to regular time series)
y = furniture['Sales'].resample('MS').mean()
print("\n\nMonth Start of 2017(Regular Time Series)")
print(y['2017':])

#Visualizing Furniture Sales Time Series Data
print("\n\n Visualizing Furniture Sales")
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
ax.set_ylabel('Furniture Sales')
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
ax.set_ylabel('Furniture Sales')

plt.legend()
plt.show()



furniture = df.loc[df['Category'] == 'Furniture']
office = df.loc[df['Category'] == 'Office Supplies']

furniture.shape, office.shape


cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)
office.drop(cols, axis=1, inplace=True)

furniture = furniture.sort_values('Order Date')
office = office.sort_values('Order Date')

furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
office = office.groupby('Order Date')['Sales'].sum().reset_index()

furniture.head()

office.head()



furniture = furniture.set_index('Order Date')
office = office.set_index('Order Date')

y_furniture = furniture['Sales'].resample('MS').mean()
y_office = office['Sales'].resample('MS').mean()

furniture = pd.DataFrame({'Order Date':y_furniture.index, 'Sales':y_furniture.values})
office = pd.DataFrame({'Order Date': y_office.index, 'Sales': y_office.values})

store = furniture.merge(office, how='inner', on='Order Date')
store.rename(columns={'Sales_x': 'furniture_sales', 'Sales_y': 'office_sales'}, inplace=True)
store.head()


plt.figure(figsize=(20, 8))
plt.plot(store['Order Date'], store['furniture_sales'], 'b-', label = 'furniture')
plt.plot(store['Order Date'], store['office_sales'], 'r-', label = 'office supplies')
plt.xlabel('Date'); plt.ylabel('Sales'); plt.title('Sales of Furniture and Office Supplies')
plt.legend();

first_date = store.ix[np.min(list(np.where(store['office_sales'] > store['furniture_sales'])[0])), 'Order Date']

print("Office supplies first time produced higher sales than furniture is {}.".format(first_date.date()))