# Laptop Price Analysis

This repository contains all the materials related to **Project 1**, which focuses on analyzing laptop prices, outlier detection, campaign day identification, and time series modeling.

---

## 📂 Project Structure
Laptop_Price_Analysis/
│
├── data/ # Dataset files (e.g., laptop_price_with_dates.csv)
├── docs/ # Project task document and final report (PDF, DOCX)
├── src/ # Python source code files
├── README.md # Project description (this file)



---

## 📝 Project Description

- **Objective:**  
  To analyze laptop prices, clean the dataset, detect unusual patterns, and forecast future price trends.

- **Methods used:**  
  - **IQR Method** → Detecting and removing outliers (due to skewness in the dataset).  
  - **Z-Score** → Outlier detection alternative (applicable in case of normal distribution).  
  - **Isolation Forest** → Identifying campaign/discount days.  
  - **Prophet Model** → Time series forecasting of laptop prices.  

---

## 📊 Dataset

- **Source:** The dataset is pre-prepared.  
- **Main columns:**  
  - `scrape_date` → Date when the data was collected.  
  - `price` → Laptop price (numeric).  
  - `brand`, `model`, `specs` → Laptop details and specifications.  

---

## 🖥️ Requirements

pandas
numpy
matplotlib
scikit-learn
prophet




