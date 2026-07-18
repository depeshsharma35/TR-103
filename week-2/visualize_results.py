import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results = pd.read_csv('forecast_results.csv')
results['Date'] = pd.to_datetime(results['Date'])
importance = pd.read_csv('feature_importance.csv')

# Set style
sns.set_theme(style="whitegrid")

# 1. Actual vs Predicted Sales
plt.figure(figsize=(12, 6))
plt.plot(results['Date'], results['Actual'], label='Actual Sales', color='blue', alpha=0.6)
plt.plot(results['Date'], results['Predicted'], label='Forecasted Sales', color='red', linestyle='--')
plt.title('Sales Forecast vs Actual Performance', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('sales_forecast_comparison.png')
plt.show()

# 2. Feature Importance for Business Insights
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance.head(10), hue='Feature', palette='viridis', legend=False)
plt.title('Key Drivers of Sales Demand', fontsize=16)
plt.xlabel('Impact Level', fontsize=12)
plt.ylabel('Business Factor', fontsize=12)
plt.tight_layout()
plt.savefig('sales_drivers.png')
plt.show()

# 3. Monthly Trend Analysis (from cleaned data)
df_cleaned = pd.read_csv('cleaned_superstore_sales.csv')
df_cleaned['Order Date'] = pd.to_datetime(df_cleaned['Order Date'])
df_cleaned['MonthYear'] = df_cleaned['Order Date'].dt.to_period('M')
monthly_sales = df_cleaned.groupby('MonthYear')['Sales'].sum().reset_index()
monthly_sales['MonthYear'] = monthly_sales['MonthYear'].astype(str)

plt.figure(figsize=(14, 7))
sns.lineplot(x='MonthYear', y='Sales', data=monthly_sales, marker='o')
plt.xticks(rotation=45)
plt.title('Historical Monthly Sales Trends', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.tight_layout()
plt.savefig('monthly_trends.png')
plt.show()

print("Visualizations created: sales_forecast_comparison.png, sales_drivers.png, monthly_trends.png")
