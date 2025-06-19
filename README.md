# Sensor Data Logger & Analyzer

## Project Overview

This project focuses on analyzing real-world sensor data collected from a semiconductor manufacturing process. It builds upon my previous project (a Maintenance Logger) and serves as a foundation for future machine learning work on predictive maintenance.

In this project, I focus on:
- Loading and inspecting raw industrial sensor data
- Cleaning missing or noisy data
- Visualizing patterns, trends, and anomalies
- Performing correlation analysis between sensors
- Preparing the data for future machine learning applications

---

## Key Concepts Covered

- Data inspection using `pandas`
- Handling missing values
- Statistical summary and sensor health analysis
- Time-based smoothing with rolling averages
- Sensor-to-sensor correlation analysis
- Data visualization with `matplotlib` and `seaborn`

---

## Dataset

The dataset used in this project is the [SECOM dataset](https://archive.ics.uci.edu/ml/datasets/secom) from the UCI Machine Learning Repository. It contains 590 sensor measurements from a semiconductor manufacturing process, with some rows labeled as pass/fail.

- `secom.csv`: Cleaned version of the original `.data` file
- `secom.names`: Description of dataset features and layout

---

## Tools Used

- Python 3.13.3
- [pandas](https://pandas.pydata.org/) for data handling
- [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) for visualization
- Git & GitHub for version control

---

## Project Progression

1. **Data Loading** — Load and inspect raw sensor data
2. **Data Cleaning** — Handle missing values and save a clean version
3. **Visualization** — Plot sensor signals, distributions, and trends
4. **Analysis** — Examine rolling averages and correlation between sensor features
5. **Packaging** — Wrap up into a usable analysis script

---

## What I Learned

- How to work with large, real-world industrial datasets
- How to explore and visualize high-dimensional data
- How to identify trends and anomalies in sensor signals
- How to prepare data for future machine learning pipelines

---

## Future Work

In a future project, I will build on this dataset to develop a predictive maintenance model using machine learning. The goal will be to detect signs of failure in advance using classification algorithms.

---

## Acknowledgments

- UCI Machine Learning Repository for the SECOM dataset
- [pandas](https://pandas.pydata.org/) and [matplotlib](https://matplotlib.org/) teams for documentation and tutorials
