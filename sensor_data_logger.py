import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Turns the CSV file into a readable dataframe
df = pd.read_csv("secom.csv")
# Fills the missing values with the average
df_filled = df.fillna(df.mean())

scaler = StandardScaler()

# Learns the means and standard deviations of the columns and standardizes all the numbers with that info
df_scaled = pd.DataFrame(scaler.fit_transform(df_filled), columns=df_filled.columns)

# Assume data was collected every 10 second starting from 9:00 AM on June 20, 2025
start_time = pd.Timestamp("2025-06-20 09:00:00")

# Creates a list of timestamps
timestamps = pd.date_range(start=start_time, periods=len(df_filled), freq="10s")

# Add timestamps as new column
df_filled["Timestamp"] = timestamps

# Creates a blank canvas that is 10 inches wide x 5 inches tall
plt.figure(figsize=(10,5))

# Plots a a line plot(x-axis: Time Values, y-axis: Sensor Readings, name of line for legend)
plt.plot(df_filled["Timestamp"], df_filled["0"], label="Sensor 0")
plt.plot(df_filled["Timestamp"], df_filled["1"], label="Sensor 1")
plt.plot(df_filled["Timestamp"], df_filled["2"], label="Sensor 2")
plt.plot(df_filled["Timestamp"], df_filled["3"], label="Sensor 3")
plt.plot(df_filled["Timestamp"], df_filled["4"], label="Sensor 4")
# Clumping plots commands together puts them on the same graph so that you can compare them

plt.xlabel("Time")
plt.ylabel("Sensor Reading")
plt.title("Sensor 0 Readings Over Time")
# Rotates the labels on the x-axis
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()