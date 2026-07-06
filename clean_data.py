import pandas as pd

try:
    df = pd.read_csv('superstore_sales.csv', encoding='ISO-8859-1')
except Exception as e:
    print(f"Error with ISO-8859-1: {e}")
    df = pd.read_csv('superstore_sales.csv', encoding='cp1252')

print("Initial Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df.columns = df.columns.str.strip()

df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')

df = df.sort_values('Order Date')

df = df.dropna(subset=['Sales', 'Order Date'])

df.to_csv('cleaned_superstore_sales.csv', index=False)
print("\nData cleaning completed. Saved to 'cleaned_superstore_sales.csv'.")
