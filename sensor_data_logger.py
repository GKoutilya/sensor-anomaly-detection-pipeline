import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("secom.csv")

print(df.head())
print(df.columns)
print(df.info())
print(df.describe())