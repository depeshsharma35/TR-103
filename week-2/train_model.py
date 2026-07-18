import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the features
df = pd.read_csv('sales_features.csv')

# Define features and target
features = ['Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'Quarter', 
            'Sales_Lag_1', 'Sales_Lag_7', 'Sales_Lag_30', 'Sales_MA_7', 'Sales_MA_30']
target = 'Sales'

X = df[features]
y = df[target]

# Time-series split (train on past, test on future)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save results for visualization
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Date': df['Order Date'].iloc[split_idx:]})
results.to_csv('forecast_results.csv', index=False)

# Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)
feature_importance_df.to_csv('feature_importance.csv', index=False)

print("\nModel training and evaluation completed.")
