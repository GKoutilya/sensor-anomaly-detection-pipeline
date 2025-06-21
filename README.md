# Project #2 – Anomaly Detection in SECOM Sensor Data

## Overview

This project focuses on **detecting anomalies** in sensor data collected from a semiconductor manufacturing process. The goal is to build a robust pipeline that cleans, visualizes, and analyzes high-dimensional sensor data using machine learning — specifically the **Isolation Forest algorithm** — and generates a polished PDF report of key findings.

---

## What This Project Does

- Reads and cleans raw sensor data (`secom.csv`)
- Fills in missing values and standardizes features
- Adds timestamps and computes rolling averages
- Visualizes trends for **all 590 sensors**
- Identifies **interesting sensors** based on:
  - Extreme values
  - Low or high standard deviation
- Cleans and deduplicates sensor data using correlation analysis
- Applies **Isolation Forest** to detect anomalies
- Visualizes anomalies with **Seaborn**
- Automatically generates a PDF anomaly report with key plots and metrics

---

## Visual Output

Plots of raw + smoothed sensor data, highlighting sensors with unusual behavior. Sample below:

![sample](sensor_plots/sample_sensor.png)

---

## Features Added

| Feature                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Feature Engineering    | Rolling averages, outlier filtering, redundancy removal                     |
| Visualizations         | Matplotlib + Seaborn plots for trend and anomaly detection                  |
| Model                  | Isolation Forest (scikit-learn) for unsupervised anomaly detection          |
| Report Generation      | Auto-created PDF summarizing anomaly scores and flagged data points         |
| Organization           | Modular phase-based pipeline: ingestion → visualization → detection → export|

---

## Tech Stack

- Python 3.13.3
- Pandas, NumPy  
- Matplotlib, Seaborn  
- scikit-learn  
- fpdf2 (for PDF reports)

---

## Getting Started

1. Place `secom.csv` in the root directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
