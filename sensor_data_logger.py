import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Turns the CSV file into a readable dataframe
df = pd.read_csv("secom.csv")
#Fills the missing values with the average
df_filled = df.fillna(df.mean())

scaler = StandardScaler()

#Learns the means and standard deviations of the columns and standardizes all the numbers with that info
df_scaled = pd.DataFrame(scaler.fit_transform(df_filled), columns=df_filled.columns)