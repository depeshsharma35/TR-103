import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_superstore_sales.csv')

df['Order Date'] = pd.to_datetime(df['Order Date'])

daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

daily_sales = daily_sales.sort_values('Order Date')

daily_sales['Year'] = daily_sales['Order Date'].dt.year
daily_sales['Month'] = daily_sales['Order Date'].dt.month
daily_sales['Day'] = daily_sales['Order Date'].dt.day
daily_sales['DayOfWeek'] = daily_sales['Order Date'].dt.dayofweek
daily_sales['IsWeekend'] = daily_sales['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
daily_sales['Quarter'] = daily_sales['Order Date'].dt.quarter

daily_sales['Sales_Lag_1'] = daily_sales['Sales'].shift(1)
daily_sales['Sales_Lag_7'] = daily_sales['Sales'].shift(7)
daily_sales['Sales_Lag_30'] = daily_sales['Sales'].shift(30)

daily_sales['Sales_MA_7'] = daily_sales['Sales'].rolling(window=7).mean()
daily_sales['Sales_MA_30'] = daily_sales['Sales'].rolling(window=30).mean()

daily_sales = daily_sales.dropna()

daily_sales.to_csv('sales_features.csv', index=False)
print("Feature engineering completed. Saved to 'sales_features.csv'.")
print(daily_sales.head())
