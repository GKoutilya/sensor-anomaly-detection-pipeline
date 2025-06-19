import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("secom.csv")

'''print(df.head())
print(df.columns)
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.dtypes)
print(df['0'].head(10))
print(df['0'].tail(10))'''

df_filled = df.fillna(df.mean())


print(df_filled.head())
print(df_filled.info())
print(df_filled.describe())

print("Hello World")