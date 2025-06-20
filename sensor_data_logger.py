import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler # Scikit-learn package - popular machine learning library
import os

# Turns the CSV file into a readable dataframe
df = pd.read_csv("secom.csv")
# Fills the missing values with the average
df_filled = df.fillna(df.mean())

# Learns the mean and standard deviation of the input data
scaler = StandardScaler()

# Learns the mean and standard deviations of the columns and standardizes all the numbers with that info
df_scaled = pd.DataFrame(scaler.fit_transform(df_filled), columns=df_filled.columns)

# Assume data was collected every 10 second starting from 9:00 AM on June 20, 2025
start_time = pd.Timestamp("2025-06-20 09:00:00")

# Creates a list of timestamps
timestamps = pd.date_range(start=start_time, periods=len(df_filled), freq="10s")

# Add timestamps as new column
df_filled["Timestamp"] = timestamps

# Checks to see if the folder already exists
output_folder = "sensor_plots"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# This whole entire loops goes through all 590 sensors (columns), plots the data and the rolling average of the data on a graph, and saves the graphs as PNG files into a folder
for sensor_num in range(590):
    print(f"Processing sensor {sensor_num} / 589...")

    filepath = os.path.join(output_folder, f"sensor_{sensor_num}.png")
    if not os.path.exists(filepath):
        # Standardizes data from each sensor
        df_filled[f"Rolling_{sensor_num}"] = df_filled[f"{sensor_num}"].rolling(window=10).mean()

        # Creates a blank canvas of size 10 inches wide x 5 inches tall
        plt.figure(figsize=(10,5))

        # Plots the raw sensor data and the standardized sensor data on a graph together
        plt.plot(df_filled["Timestamp"], df_filled[f"{sensor_num}"], label=f"Sensor {sensor_num} (Raw)")
        plt.plot(df_filled["Timestamp"], df_filled[f"Rolling_{sensor_num}"], label=f"Sensor {sensor_num} (Rolling Avg)")

        # Organizes the graphs
        plt.xlabel("Time")
        plt.ylabel("Sensor Reading")
        plt.title(f"Sensor {sensor_num} Readings Over Time")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        # Save the graph as a PNG to file
        plt.savefig(filepath)

        # Close the plot (important to prevent memory issues)
        plt.close()
    else:
        print(f"Sensor {sensor_num} already has a saved plot. Skipping.")

# Drops Timestamp when looking at the statistics and describes the sensor statistics
sensor_data_only = df_filled.drop(columns=["Timestamp"])
summary_stats = sensor_data_only.describe()

# Sensors with very low standard deviation (flat sensors)
low_std_sensors = summary_stats.loc["std"][summary_stats.loc["std"] < 0.01]
print("Very flat sensors (low std):")
print(low_std_sensors)

# Sensors with very high standard deviation (noisy sensors)
high_std_sensors = summary_stats.loc["std"][summary_stats.loc["std"] > 1000]
print("Very noisy sensors (high std):")
print(high_std_sensors)

# Sensors with extreme maximum values
weird_max = summary_stats.loc["max"][summary_stats.loc["max"] > 10000]
print("Sensors with extremely high max values:")
print(weird_max)

# Sensors with extreme minimum values
weird_min = summary_stats.loc["min"][summary_stats.loc["min"] < -10000]
print("Sensors with extremely low min values:")
print(weird_min)

# Combine all interesting sensors into one set to avoid duplicates
interesting_sensors = set(low_std_sensors.index) | set(high_std_sensors.index) | set(weird_max.index) | set(weird_min.index)

# Creates a new folder for all of these interesting sensors, if one isn't already created
interesting_folder = "interesting_sensor_plots"
if not os.path.exists(interesting_folder):
    os.makedirs(interesting_folder)

# Make sure rolling averages exist for all sensors
for sensor_num in range(590):
    col_name = f"Rolling_{sensor_num}"
    if col_name not in df_filled.columns:
        df_filled[col_name] = df_filled[f"{sensor_num}"].rolling(window=10).mean()

# The loop goes through all of the interesting sensors and graphs the raw data and the standardized data on the same graph and saves a PNG file of the graph to a specific folder
for sensor_num in interesting_sensors:
    print(f"Plotting interesting sensor {sensor_num}")

    # This creates a path between what should be saved (the PNG files) and where it should be saved (the folder called intersting_folder)
    filepath = os.path.join(interesting_folder, f"sensor_{sensor_num}.png")

    # The if-statement checks to make sure that the PNG files aren't already saved prior
    if not os.path.exists(filepath):
        # This creates the graph
        plt.figure(figsize=(10,5))

        # These two lines plots the raw data and the standardized data on the graph
        plt.plot(df_filled["Timestamp"], df_filled[sensor_num], label=f"Sensor {sensor_num} (Raw)")
        plt.plot(df_filled["Timestamp"], df_filled[f"Rolling_{sensor_num}"], label=f"Sensor {sensor_num} (Rolling Avg)")

        # These lines organizes the graph
        plt.xlabel("Time")
        plt.ylabel("Sensor Reading")
        plt.title(f"Sensor {sensor_num} Readings Over Time")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        # The PNG files are saved and the graphs are closed
        plt.savefig(filepath)
        plt.close()
    else:
        print(f"Sensor {sensor_num} already plotted. Skipping.")

# Flat sensor data with low standard deviation is uninformative and should be removed, df_cleaned is the new dataset without flat/uninformative sensors
uninformative_sensors = list(low_std_sensors.index)
df_cleaned = df_filled.drop(columns=uninformative_sensors)

# Drop all columns that start with "Rolling_"
df_cleaned = df_filled.drop(columns=[col for col in df_filled.columns if col.startswith("Rolling_")])


# The Timestamp column that we added gets in the way so we need to remove it to have only sensor data
sensor_data_cleaned = df_cleaned.drop(columns=["Timestamp"])


# A Correlation Matrix is a table/matrix that correlates every variable in a dateset (in this case sensors) to the others
# Correlation occurs on a scale from -1 to 1
# +1 means there is a perfect positive correlation between to variables: when ones goes up, the other goes up the exact same way
# -1 means there is a perfect negative correlation between to variables: when ones goes up, the other goes down the exact same way
# 0 means there is no linear relationship between the variables

# This gives us the correlation matrix of the cleaned up sensor data.
# abs() finds the absolute value of the correlation matrix, we don't care if the correlation is positive or negative, as long as there is one
corr_matrix = sensor_data_cleaned.corr().abs()

# We don't want any duplicate values so we retrieve only half of the correlation matrix and discard the rest of the repeated values
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# This creates a list of variables (sensors) that have a very high correlation between them that we can observe
# We create this list to drop because sensors that are that highly correlated might measure the same thing so they are redundant to keep
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# This drops the redundant list from the already "cleaned" dataset to give use our final dataset that we can use and study
df_final = df_cleaned.drop(columns=to_drop)


# This saves our final, cleaned dataset to a CSV file that we can look at
if not os.path.exists("secom_cleaned.csv"):
    df_final.to_csv("secom_cleaned.csv", index=False)
    print("Cleaned dataset saved as 'secom_cleaned.csv'")
else:
    print("File 'secom_cleaned.csv' already exists. Skipping save.")

print("Original shape:", df.shape)
print("After cleaning:", df_final.shape)
