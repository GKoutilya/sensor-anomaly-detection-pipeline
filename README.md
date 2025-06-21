# Project #2 – Anomaly Detection in SECOM Sensor Data

## Overview

This project focuses on **detecting anomalies** in high-dimensional sensor data from a semiconductor manufacturing process (SECOM dataset). It builds a complete data pipeline that performs preprocessing, exploratory visualization, statistical filtering, machine learning–based anomaly detection, and generates a PDF report of key insights.

---

## Key Functionality

- Loads and cleans raw data from `secom.csv`
- Handles missing values and standardizes sensor readings
- Adds synthetic timestamps and computes rolling averages
- Visualizes trends for **all 590 sensors**
- Identifies **noteworthy sensors** based on:
  - Extremely high/low values
  - Low or high standard deviation
- Reduces redundancy using correlation-based filtering
- Applies **Isolation Forest** to detect anomalous patterns
- Creates visualizations of anomalies using **Seaborn**
- Automatically generates a PDF anomaly report with visuals and metrics

---

## Output Example

Line plots showing raw vs. smoothed sensor values, with anomalies highlighted.  
Sample preview:

![sample](sensor_plots/sample_sensor.png)

---

## Features

| Feature                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **Feature Engineering**| Adds rolling averages, filters flat/noisy signals, removes redundancy       |
| **Visualization**      | Time-series plots, correlation heatmaps, anomaly score distributions        |
| **Modeling**           | Uses Isolation Forest (unsupervised) to identify anomalous data points      |
| **Reporting**          | Automatically creates a PDF summarizing findings with embedded plots        |
| **Modular Design**     | Structured in 4 clearly separated phases for scalability and readability     |

---

## 🛠 Tech Stack

- **Python** 3.13.3  
- **Pandas**, **NumPy** – Data handling  
- **Matplotlib**, **Seaborn** – Visualizations  
- **scikit-learn** – Machine learning (Isolation Forest)  
- **fpdf2** – PDF report generation

---

## Getting Started

1. Clone the repository and place `secom.csv` in the root directory.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
4. Run the main script:
    python sensor_data_logger.py

---

## Notes

The SECOM dataset has many missing values and correlated features; preprocessing is critical.

Isolation Forest is particularly suited to high-dimensional, unlabeled data like this.

Run time may vary depending on system performance due to the number of plots.

---

## Author

- Koutilya Ganapathiraju
- Texas A&M University - College Station
- Manufacturing & Mechanical Engineering