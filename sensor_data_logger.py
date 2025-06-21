import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler # Scikit-learn package - popular machine learning library
from sklearn.ensemble import IsolationForest
from datetime import datetime
from fpdf import FPDF

# Loads and standardizes the raw sensor data, filling in missing values and adding timestamps.
def phase1():
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

    return df_filled

# Generates and saves time-series plots (raw and rolling average) for all sensor readings.
def phase2(df_filled):
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

# Indentifies and removes uninformative or redundant sensors based on statistics and correlation, saving a cleaned dataset.
def phase3(df_filled):
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
    df_cleaned = df_cleaned.drop(columns=[col for col in df_cleaned.columns if col.startswith("Rolling_")])



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

    print("Original shape:", df_filled.shape)
    print("After cleaning:", df_final.shape)

    return df_final

# Detects anomalies using Isolation Forest, visualizes key results, and generates a summmary PDF report.
def phase4(df_final):
    # Copy cleaned dataset
    df = df_final.copy()

    # Remove Timestamp or any non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    # Check no missing values
    assert not df_numeric.isnull().values.any(), "Data contains missing values."

    # Isolation Forest is ideal for high-dimensional data like sensors, chose it because it performs well for unsupervised anomaly detection, is efficient, and doesn't require labeled data.

    # Create an Isolation Forest model, more trees = better but slower
    # Isolation Forests find anomalies by recursively partitioning data into random subsets (trees).
    # Data points that are isolated quickly (with fewer splits, i.e., shorter paths) are considered anomalies because they differ significantly from the majority of the data.
    # n_estimators - number of trees in the forest
    # contamination - estimated fraction of anomalies in data which helps model decide threshold
    # max_samples - number of samples to draw from data to train each tree, default is auto
    # random_state - to makes ure you get same results every time you run it
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

    # Trains the model on the data
    model.fit(df_numeric)

    # Predict anomalies (-1 = anomaly, 1 = normal) and add as new column
    df["Anomaly"] = model.predict(df_numeric)

    # .decision_function(x) -  gives a score, lower score means more anomalous
    # Add anomaly scores
    df["Anomaly_Score"] = model.decision_function(df_numeric)

    # Count how many anomalies were found
    num_anomalies = (df["Anomaly"] == -1).sum()
    print(f"Number of anomalies detected: {num_anomalies} out of {len(df)} rows")

    # Save anomalies for later review after checking to see if it already exists
    if not os.path.exists("anomalies_detected.csv"):
        df[df["Anomaly"] == -1].to_csv("anomalies_detected.csv", index=False)
        print("Anomalies saved to 'anomalies_detected.csv'")
    else:
        print("'anomalies_detected.csv' already exists. Skipping save.")

    visualize_anomalies(df)
    generate_pdf(df)

    return df

def visualize_anomalies(df):
    # Checks to see if a folder called "anomaly_visuals" exists and creates it if it doesn't exist
    output_folder = "anomaly_visuals"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Selects all numeric columns that represent actual sensor data and not the columns that were added during anomaly detection: "Anomaly" and "Anomaly_Score"
    numeric_columns = df.select_dtypes(include=[np.number]).columns.drop(["Anomaly", "Anomaly_Score"], errors='ignore')

    # Counts how many rows are labeled as anomalies for each sensor and stores them in a dictionary
    anomaly_counts = {
    col: df[(df["Anomaly"] == -1) & (df[col].notnull())].shape[0]
    for col in numeric_columns
    }

    # Adds the top 5 most and least anomalous sensors to a list
    most_anomalous = sorted(anomaly_counts, key=anomaly_counts.get, reverse=True)[:5]
    least_anomalous = sorted(
        [k for k, v in anomaly_counts.items() if v > 0],
        key=anomaly_counts.get
    )[:5]

    selected_sensors = most_anomalous + least_anomalous
    print("Most anomalous sensors: ", most_anomalous)
    print("Least anomalous sensors: ", least_anomalous)
 
    # For each sensor in the combined list:
    for sensor in selected_sensors:
        filepath = os.path.join(output_folder, f"anomaly_plot_{sensor}.png")
        if not os.path.exists(filepath):
            plt.figure(figsize=(10,5))
            # Creates a line plot of the sensor readings over all samples
            sns.lineplot(x=range(len(df)), y=df[sensor], label="Sensor Reading")
            # Overlays red scatter points where anomalies are detected
            sns.scatterplot(x=df[df["Anomaly"] == -1].index, y=df[df["Anomaly"] == -1][sensor], color='red', label='Anomaly', s=50)
            # Titles the plot
            plt.title(f"{'Most' if sensor in most_anomalous else 'Least'} Anomalous Sensor: {sensor}")
            plt.xlabel("Sample Index")
            plt.ylabel("Sensor Value")
            plt.legend()
            plt.tight_layout()
            # Saves a PNG file under the "anomaly_visuals" folder (only if the plot doesn't already exist)
            plt.savefig(filepath)
            plt.close()

    # If the file doesn't already exist:
    corr_path = os.path.join(output_folder, "correlation_heatmap.png")
    if not os.path.exists(corr_path):
        # Creates a correlation matrix between all numeric sensor columns
        corr = df[numeric_columns].corr()
        plt.figure(figsize=(12,10))
        # Plots the correlation matrix as a heatmap
        sns.heatmap(corr, cmap='coolwarm', center=0)
        plt.title("Sensor Correlation Heatmap")
        plt.tight_layout()
        # Saves the heatmap as a PNG file
        plt.savefig(corr_path)
        plt.close()

    # If the file doesn't already exist
    score_path = os.path.join(output_folder, "anomaly_score_distribution.png")
    if not os.path.exists(score_path):
        plt.figure(figsize=(8, 5))
        # Plots a histogram of the anomaly scores - continuous values that indicate the degree of anomaly
        # Also includes a KDE curve (a smoothed estimate) for distribution shape
        sns.histplot(df["Anomaly_Score"], bins=50, kde=True)
        plt.title("Anomaly Score Distribution")
        plt.xlabel("Score")
        plt.tight_layout()
        # Saves the histogram as a PNG file
        plt.savefig(score_path)
        plt.close()

    # If the file doesn't already exist
    bar_path = os.path.join(output_folder, "sensor_anomaly_counts.png")
    if not os.path.exists(bar_path):
        anomaly_series = pd.Series(anomaly_counts).sort_values(ascending=False)
        plt.figure(figsize=(12,6))
        # Creates a bar plot showing the number of anomalies for every sensor
        # Sensors are sorted descending by anomaly count
        sns.barplot(x=anomaly_series.index, y=anomaly_series.values)
        # Sensor names on the x-axis are rotated for readability
        plt.xticks(rotation=90)
        plt.title("Number of Anomalies per Sensor")
        plt.xlabel("Sensor")
        plt.ylabel("Anomaly Count")
        plt.tight_layout()
        # Saves the bar plot as a PNG file
        plt.savefig(bar_path)
        plt.close()

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(0, 10, "Hello, world!")

class PDFWithPageNumbers(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        page_number = f"Page {self.page_no()}"
        self.cell(0, 10, page_number, 0, 0, "C")

def generate_pdf(df):
    # Checks to see whether the PDF already exists or not, skips the whole process if it does
    pdf_path = "anomaly_report.pdf"
    if os.path.exists(pdf_path):
        print("PDF report already exists. Skipping generation.")
        return
    
    # Creates a new PDF object from a subclass that adds page numbers in the footer
    pdf = PDFWithPageNumbers()
    # Enables automatic page breaks with a 15mm margin
    pdf.set_auto_page_break(auto=True, margin=15)
    # Adds the first page and sets the default font and size
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", 'B', 16)
    # Adds a centered bold title
    pdf.cell(200,10,txt="Anomaly Detection Report", ln=True, align='C')
    # Prints a timestamp in italics showing the time the report was generated
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Generated on: {timestamp}", ln=True)
    pdf.set_font("Arial", size=12)
    # Adds vertical spacing?
    pdf.ln(10)

    # Shows key statistics such as total number of samples and how many anomalies were detected
    total_rows = len(df)
    total_anomalies = (df["Anomaly"] == -1).sum()
    pdf.cell(0,10, f"Total data points: {total_rows}", ln=True)
    pdf.cell(0,10, f"Anomalies detected: {total_anomalies}", ln=True)
    # Adds a short description about what's to follow in the report
    pdf.cell(0,10, "The next pages show comparisons between the most and least anomalous sensors.", ln=True)

    # This whole chunk of code iterates over all the PNG files in the anomaly_visuals folder and adds each image to its own page and scales the width to 180mm
    image_folder = "anomaly_visuals"
    for img_file in sorted(os.listdir(image_folder)):
        if img_file.endswith(".png"):
            pdf.add_page()
            pdf.image(os.path.join(image_folder, img_file), w=180)
    
    # This saves the PDF file to a disk and prints and output message to the console
    pdf.output(pdf_path)
    print("PDF report saved as 'anomaly_report.pdf'")

##################### MAIN ########################

if __name__ == "__main__":
    df_filled = phase1()
    phase2(df_filled)
    df_final = phase3(df_filled)
    df = phase4(df_final)