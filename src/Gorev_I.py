# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 14:51:26 2025

@author: Şamil Bozkurt
"""






import numpy as np 
import pandas as pd


#---------------------------------
# Read data and view the first 5 rows
# The dataset is pre-prepared and random dates have been added.

df = pd.read_csv("laptop_price_with_dates.csv")
df.head()
#---------------------------------

#---------------------------------
df.info()

# The "scrape_date" column is of type object, it needs to be converted to datetime.
# For time series modeling, it must be datetime.
#---------------------------------

#---------------------------------
df["scrape_date"] = pd.to_datetime(df["scrape_date"])

# Convert the "scrape_date" column to datetime.
df.info()
#---------------------------------

#---------------------------------
df.isnull().any()

# Are there any missing values in the dataset?
# It is observed that there are no missing values.
#---------------------------------

#---------------------------------
df.describe().T

# Check the basic statistical information of numerical variables
#---------------------------------

#---------------------------------
# For statistical information of numerical data, "laptop_ID" and "scrape_date" columns are not important.
df[["Inches","Price_euros"]].describe().T

# First noticeable point --> A visible difference in prices (difference between 3rd quartile and max price).
# Considering the average price, this comment was made.
# There is a high possibility that the max price is an outlier.
#---------------------------------

#---------------------------------
# Calculating values using the IQR method to detect and remove outliers
# Find the 1st and 3rd quartiles
Q1 = df["Price_euros"].quantile(0.25)
Q3 = df["Price_euros"].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Determine lower and upper limits
alt_sinir = Q1 - 1.5 * IQR
ust_sinir = Q3 + 1.5 * IQR

# Detect outliers
outliers = df[(df["Price_euros"] < alt_sinir) | (df["Price_euros"] > ust_sinir)]


# Print results (view first 10 outliers)
print(f"Number of outliers: {len(outliers)}")
print(f"Lower limit: {alt_sinir:.2f}, Upper limit: {ust_sinir:.2f}")
outliers[["Company", "Product", "Price_euros"]].head(10)
#---------------------------------

#---------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 2))
sns.boxplot(x=df["Price_euros"])
plt.title("Boxplot of Prices (Outliers with IQR)")

# 📥 Save as PNG 
plt.savefig("boxplot_fiyat_iqr.png", dpi=300, bbox_inches='tight')

plt.show()
#---------------------------------

#---------------------------------
# Calculating values using the Z-Score method to detect and remove outliers
from scipy.stats import zscore

# Calculate Z-Score
df['z_score'] = zscore(df["Price_euros"])

# Mark the outliers
df['is_outlier'] = df['z_score'].abs() > 3

# Filter the outliers
outliers = df[df['is_outlier'] == True]

# View the results
print("Number of outliers:", len(outliers))
print(outliers[["Price_euros", 'z_score']])
#---------------------------------

#---------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.hist(df['z_score'], bins=50, color='skyblue')
plt.axvline(x=3, color='red', linestyle='--')
plt.axvline(x=-3, color='red', linestyle='--')
plt.title("Z-Score Distribution")
plt.xlabel("Z-Score")
plt.ylabel("Frequency")
plt.show()
#---------------------------------

#---------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(df['Price_euros'], bins=30, kde=True, color='skyblue')
plt.title("Price Distribution")
plt.xlabel("Price (€)")
plt.ylabel("Frequency")
plt.savefig("fiyat_dagilim.png", dpi=300, bbox_inches='tight')
plt.show()
#---------------------------------

#---------------------------------
# It is observed that the dataset is skewed. Therefore, the IQR method is more reasonable. If it had a normal distribution, we could have chosen the Z-Score method.
df_cleaned = df[(df["Price_euros"] >= alt_sinir) & (df["Price_euros"] <= ust_sinir)]
df_cleaned.describe().T

# For price evaluation - we eliminated the outliers and observed it statistically.
#---------------------------------

#---------------------------------
# Prophet model for time series modeling.
# Required libraries
from prophet import Prophet
import matplotlib.pyplot as plt


# Transformation required for Prophet
df_prophet = df_cleaned[["scrape_date", "Price_euros"]].rename(columns={
    "scrape_date": "ds",
    "Price_euros": "y"
})

df_prophet["y"] = np.log(df_prophet["y"])


# Build and train Prophet model
model = Prophet()
model.fit(df_prophet)

# Generate forecast for the future (e.g. 60 days)
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# Visualize forecasts
fig1 = model.plot(forecast)
plt.title("Laptop Price Forecast - Prophet")
plt.xlabel("Date")
plt.ylabel("Price (€)")
plt.show()

fig1.savefig("prophet_forecast.png", dpi=300, bbox_inches='tight')
#---------------------------------

#---------------------------------
fig2 = model.plot_components(forecast)
fig2.savefig("prophet_components.png", dpi=300, bbox_inches='tight')
#---------------------------------

#---------------------------------
df_daily = df_cleaned.groupby('scrape_date')['Price_euros'].mean().reset_index()
df_daily.columns = ['date', 'price']
df_daily.set_index('date', inplace=True)
df_daily.head()
#---------------------------------

#---------------------------------
# Using Isolation Forest and Z-Score methods to detect campaign days.

from sklearn.ensemble import IsolationForest

# Build Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)

# Modeling based on price
df_daily['anomaly_iforest'] = model.fit_predict(df_daily[['price']])

# Anomaly = -1, Normal = 1
df_anomalies_if = df_daily[df_daily['anomaly_iforest'] == -1]
df_anomalies_if.head()
#---------------------------------

#---------------------------------
from scipy.stats import zscore

# Calculate Z-score
df_daily['z_score'] = zscore(df_daily['price'])

# Values smaller than -1.8 (sudden price drops)
df_daily['anomaly_zscore'] = df_daily['z_score'] < -1.8

# Anomaly days (True values)
df_anomalies_z = df_daily[df_daily['anomaly_zscore'] == True]
df_anomalies_z.head()
#---------------------------------

#---------------------------------
# Extract date sets
dates_iforest = set(df_anomalies_if.index)
dates_zscore = set(df_anomalies_z.index)

# Common days
common_dates = dates_iforest.intersection(dates_zscore)

# Days specific to Isolation Forest
only_iforest = dates_iforest - dates_zscore

# Days specific to Z-score
only_zscore = dates_zscore - dates_iforest

# Summary
print("Number of days detected by Isolation Forest:", len(dates_iforest))
print("Number of days detected by Z-score:", len(dates_zscore))
print("Number of common days detected by both methods:", len(common_dates))
#---------------------------------

#---------------------------------
# Days labeled as ‘Both’ → Can be reliably reported as campaign/discount days.

# Prices of common days
df_common = df_daily.loc[df_daily.index.isin(common_dates)].copy()
df_common['method'] = 'Both'

# Only Isolation Forest
df_only_if = df_daily.loc[df_daily.index.isin(only_iforest)].copy()
df_only_if['method'] = 'Only Isolation Forest'

# Only Z-score
df_only_z = df_daily.loc[df_daily.index.isin(only_zscore)].copy()
df_only_z['method'] = 'Only Z-score'

# Merge all
df_comparison = pd.concat([df_common, df_only_if, df_only_z])
df_comparison = df_comparison.sort_index()

# Simplify columns
df_comparison = df_comparison[['price', 'method']]
df_comparison
#---------------------------------

#---------------------------------
# Prepare product-based price forecasts for Prophet output
product_groups = df_cleaned.groupby('Product')

rapor_listesi = []

from prophet import Prophet

for product_name, group in product_groups:
    # Skip if the group is small (e.g. < 10 days)
    if group['scrape_date'].nunique() < 10:
        continue

    # Convert to Prophet format
    df_p = group.groupby('scrape_date')['Price_euros'].mean().reset_index()
    df_p.columns = ['ds', 'y']

    # Build Prophet model
    model = Prophet()
    model.fit(df_p)

    # Forecast 30 days ahead
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Past and forecast averages
    past_30_avg = df_p['y'].tail(30).mean()
    next_30_avg = forecast['yhat'].tail(30).mean()

    # Percentage change
    change_pct = 100 * (next_30_avg - past_30_avg) / past_30_avg

    rapor_listesi.append({
        'Product': product_name,
        'Last 30 Days Avg (€)': round(past_30_avg, 2),
        'Forecast 30 Days Avg (€)': round(next_30_avg, 2),
        'Percentage Change': round(change_pct, 2),
        'Comment': f"Price {'may decrease' if change_pct < 0 else 'may increase'} (%{abs(round(change_pct, 2))})"
    })


import pandas as pd
df_rapor = pd.DataFrame(rapor_listesi)
df_rapor.sort_values(by='Percentage Change').head(10)
#---------------------------------

#---------------------------------
import matplotlib.pyplot as plt

# Sort by negative changes
df_plot = df_rapor.sort_values(by='Percentage Change').head(15)

# Graphics
plt.figure(figsize=(12, 6))
bars = plt.barh(df_plot['Product'], df_plot['Percentage Change'], color=['red' if x < 0 else 'green' for x in df_plot['Percentage Change']])
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel('Predicted Percentage Change (%)')
plt.title('Predicted Product-Based Price Change for Next 30 Days')

# Write the percentages on top of the bars
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
             f"%{bar.get_width():.1f}",
             va='center', ha='left' if bar.get_width() > 0 else 'right',
             fontsize=9)

plt.tight_layout()
plt.show()
#---------------------------------
