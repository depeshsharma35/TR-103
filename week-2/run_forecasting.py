import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Phase 1: Data Cleaning ---")
try:
    df = pd.read_csv('superstore_sales.csv', encoding='ISO-8859-1')
except Exception:
    df = pd.read_csv('superstore_sales.csv', encoding='cp1252')

df.columns = df.columns.str.strip()
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df = df.sort_values('Order Date')
df = df.dropna(subset=['Sales', 'Order Date'])
print(f"Data cleaned. Total records: {len(df)}")

print("\n--- Phase 2: Feature Engineering ---")
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
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
print(f"Features engineered. Total records for training: {len(daily_sales)}")

print("\n--- Phase 3: Model Training ---")
features = ['Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'Quarter', 
            'Sales_Lag_1', 'Sales_Lag_7', 'Sales_Lag_30', 'Sales_MA_7', 'Sales_MA_30']
target = 'Sales'
X = daily_sales[features]
y = daily_sales[target]
split_idx = int(len(daily_sales) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Model Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

print("\n--- Phase 4: Visualization ---")
sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 6))
plt.plot(daily_sales['Order Date'].iloc[split_idx:], y_test, label='Actual Sales', color='blue', alpha=0.6)
plt.plot(daily_sales['Order Date'].iloc[split_idx:], y_pred, label='Forecasted Sales', color='red', linestyle='--')
plt.title('Sales Forecast vs Actual Performance', fontsize=16)
plt.legend()
plt.savefig('my_forecast_comparison.png')
print("Saved: my_forecast_comparison.png")
plt.show()

importances = model.feature_importances_
feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp.head(10), hue='Feature', palette='viridis', legend=False)
plt.title('Key Drivers of Sales Demand', fontsize=16)
plt.savefig('my_sales_drivers.png')
print("Saved: my_sales_drivers.png")
plt.show()

print("\n--- Execution Complete ---")
