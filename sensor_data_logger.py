import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

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

output_folder = "sensor_plots"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# This whole entire loops goes through all 590 sensors (columns), plots the data and the rolling average of the data on a graph, and saves the graphs as PNG files into a folder
for sensor_num in range(590):
    print(f"Processing sensor {sensor_num} / 589...")
    df_filled[f"Rolling_{sensor_num}"] = df_filled[f"{sensor_num}"].rolling(window=10).mean()

    plt.figure(figsize=(10,5))

    plt.plot(df_filled["Timestamp"], df_filled[f"{sensor_num}"], label=f"Sensor {sensor_num} (Raw)")
    plt.plot(df_filled["Timestamp"], df_filled[f"Rolling_{sensor_num}"], label=f"Sensor {sensor_num} (Rolling Avg)")

    plt.xlabel("Time")
    plt.ylabel("Sensor Reading")
    plt.title(f"Sensor {sensor_num} Readings Over Time")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save the figure to file
    plt.savefig(os.path.join(output_folder, f"sensor_{sensor_num}.png"))

    # Close the plot (important to prevent memory issues)
    plt.close()

