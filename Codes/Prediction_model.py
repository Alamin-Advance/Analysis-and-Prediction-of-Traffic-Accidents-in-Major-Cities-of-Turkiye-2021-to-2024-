"""# Prediction model using AI method

# **data preprocessing**
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Ankara_j_total_acc.csv'
data = pd.read_csv(file_path)

# Step 2: Handle missing values
data = data.interpolate(method='linear', axis=0)

# Step 3: Feature engineering
# Convert 'Month' to categorical with correct order
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
data['Month'] = pd.Categorical(data['Month'], categories=month_order, ordered=True)

# Encode 'Month' to numerical
data['Month_num'] = data['Month'].cat.codes

# Step 4: Normalize numerical features
scaler = MinMaxScaler()
numerical_columns = ['Total_Accidents', 'Death', 'Injured']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Step 5: Encode categorical features
data = pd.get_dummies(data, columns=['Year'], prefix='Year')

# Step 6: Split data into features and targets
X = data.drop(columns=['Death', 'Injured', 'Total_Accidents'])
y_accidents = data['Total_Accidents']
y_deaths = data['Death']
y_injuries = data['Injured']

# Split data into training and testing sets
X_train, X_test, y_train_accidents, y_test_accidents = train_test_split(X, y_accidents, test_size=0.2, random_state=42)
_, _, y_train_deaths, y_test_deaths = train_test_split(X, y_deaths, test_size=0.2, random_state=42)
_, _, y_train_injuries, y_test_injuries = train_test_split(X, y_injuries, test_size=0.2, random_state=42)

# Step 7: Train and evaluate models
def train_and_evaluate_model(X_train, X_test, y_train, y_test, target_name):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{target_name} Model:')
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'R²: {r2}\n')

# Accidents model
train_and_evaluate_model(X_train, X_test, y_train_accidents, y_test_accidents, 'Accidents')

# Deaths model
train_and_evaluate_model(X_train, X_test, y_train_deaths, y_test_deaths, 'Deaths')

# Injuries model
train_and_evaluate_model(X_train, X_test, y_train_injuries, y_test_injuries, 'Injuries')

# Step 8: Save the processed dataset
processed_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Ankara_j_pro1.csv'
data.to_csv(processed_file_path, index=False)

import pandas as pd

# File path for the original dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Ankara_j_total_acc.csv'

# Load the dataset
data = pd.read_csv(file_path)

# Drop the 'Month' and 'Year' columns
data = data.drop(columns=['Month', 'Year'])

# Save the updated dataset back to the same file or a new file
output_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Ankara_j_total_acc_pro.csv'
data.to_csv(output_file_path, index=False)

print(f"Columns 'Month' and 'Year' have been removed. Updated dataset saved to: {output_file_path}")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# Load the dataset
dataset_path = "/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Ankara_j_total_acc.csv"
dataset = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print("Original Dataset:")
print(dataset.head())

# Step 1: Handle missing values
# Check for missing values
print("\nMissing Values in Each Column:")
print(dataset.isnull().sum())

# Fill missing values with column mean
dataset.fillna(dataset.mean(), inplace=True)

# Step 2: Remove target or irrelevant columns (if any)
if 'month' in dataset.columns:  # Replace 'Type' with the actual target column name if needed
    dataset = dataset.drop(columns=['month'])

# Step 3: Remove outliers
# Calculate z-scores for all features
z_scores = np.abs(zscore(dataset))

# Remove rows with z-scores greater than 3 (adjust threshold as needed)
dataset = dataset[(z_scores < 3).all(axis=1)]

# Step 4: Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(dataset)

# Convert back to DataFrame for readability
processed_data = pd.DataFrame(data_normalized, columns=dataset.columns)

# Save the preprocessed data to a CSV file
output_path = "/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Ankara_j_pro2.csv"
processed_data.to_csv(output_path, index=False)
print(f"\nPreprocessed dataset saved to: {output_path}")

# Display summary statistics of the preprocessed data
print("\nSummary Statistics of Preprocessed Data:")
print(processed_data.describe())

"""**RandomForestRegressor model**"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_total_acc.csv')

# Prepare the features (X) and target variables (y)
X = data[['Death_and_injury_accidents', 'Property_damage_accidents', 'Death', 'Injured']]  # Features

# Separate target variables for accidents, deaths, and injuries
y_accidents = data['Total_Accidents']
y_deaths = data['Death']
y_injuries = data['Injured']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train_accidents, y_test_accidents = train_test_split(X, y_accidents, test_size=0.2, random_state=42)
_, _, y_train_deaths, y_test_deaths = train_test_split(X, y_deaths, test_size=0.2, random_state=42)
_, _, y_train_injuries, y_test_injuries = train_test_split(X, y_injuries, test_size=0.2, random_state=42)

# Initialize RandomForestRegressor models for each target variable
model_accidents = RandomForestRegressor(n_estimators=100, random_state=42)
model_deaths = RandomForestRegressor(n_estimators=100, random_state=42)
model_injuries = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the models
model_accidents.fit(X_train, y_train_accidents)
model_deaths.fit(X_train, y_train_deaths)
model_injuries.fit(X_train, y_train_injuries)

# Make predictions for each model on the test set
y_pred_accidents = model_accidents.predict(X_test)
y_pred_deaths = model_deaths.predict(X_test)
y_pred_injuries = model_injuries.predict(X_test)

# Evaluate the models' performance
mae_accidents = mean_absolute_error(y_test_accidents, y_pred_accidents)
mse_accidents = mean_squared_error(y_test_accidents, y_pred_accidents)
r2_accidents = r2_score(y_test_accidents, y_pred_accidents)

mae_deaths = mean_absolute_error(y_test_deaths, y_pred_deaths)
mse_deaths = mean_squared_error(y_test_deaths, y_pred_deaths)
r2_deaths = r2_score(y_test_deaths, y_pred_deaths)

mae_injuries = mean_absolute_error(y_test_injuries, y_pred_injuries)
mse_injuries = mean_squared_error(y_test_injuries, y_pred_injuries)
r2_injuries = r2_score(y_test_injuries, y_pred_injuries)

# Print the evaluation metrics for each model
print("Accidents Model:")
print(f'MAE: {mae_accidents}, MSE: {mse_accidents}, R²: {r2_accidents}')

print("\nDeaths Model:")
print(f'MAE: {mae_deaths}, MSE: {mse_deaths}, R²: {r2_deaths}')

print("\nInjuries Model:")
print(f'MAE: {mae_injuries}, MSE: {mse_injuries}, R²: {r2_injuries}')

# Predictions for a specific test set (Optional)
print("\nPredictions for a specific test set (Accidents):")
print(y_pred_accidents[:10])  # Show first 10 predictions for accidents

print("\nPredictions for a specific test set (Deaths):")
print(y_pred_deaths[:10])  # Show first 10 predictions for deaths

print("\nPredictions for a specific test set (Injuries):")
print(y_pred_injuries[:10])  # Show first 10 predictions for injuries

import numpy as np
import matplotlib.pyplot as plt

# Plotting the predicted vs actual values for each model using line charts

# Line chart for Total Accidents
plt.figure(figsize=(7, 5))
plt.plot(y_test_accidents.values, label='Actual Accidents', color='blue')
plt.plot(y_pred_accidents, label='Predicted Accidents', color='red', linestyle='--')
plt.title('Actual vs Predicted Accidents', fontsize=10)
plt.xlabel('Data Points', fontsize=10)
plt.ylabel('Total Accidents', fontsize=10)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Line chart for Total Deaths
plt.figure(figsize=(7, 5))
plt.plot(y_test_deaths.values, label='Actual Deaths', color='blue')
plt.plot(y_pred_deaths, label='Predicted Deaths', color='red', linestyle='--')
plt.title('Actual vs Predicted Deaths', fontsize=10)
plt.xlabel('Data Points', fontsize=10)
plt.ylabel('Total Deaths', fontsize=10)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Line chart for Total Injuries
plt.figure(figsize=(7, 5))
plt.plot(y_test_injuries.values, label='Actual Injuries', color='blue')
plt.plot(y_pred_injuries, label='Predicted Injuries', color='red', linestyle='--')
plt.title('Actual vs Predicted Injuries', fontsize=10)
plt.xlabel('Data Points', fontsize=10)
plt.ylabel('Total Injuries', fontsize=10)
plt.legend()
plt.tight_layout()
plt.show()

# Function for plotting bar charts
def plot_bar_chart(actual, predicted, title, ylabel):
    indices = np.arange(len(actual))
    bar_width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(indices, actual, bar_width, label='Actual', color='blue')
    plt.bar(indices + bar_width, predicted, bar_width, label='Predicted', color='red', alpha=0.7)
    plt.title(title, fontsize=10)
    plt.xlabel('Data Points', fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.legend()
    #plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Bar chart for Total Accidents
plot_bar_chart(y_test_accidents.values, y_pred_accidents, 'Actual vs Predicted Accidents', 'Total Accidents')

# Bar chart for Total Deaths
plot_bar_chart(y_test_deaths.values, y_pred_deaths, 'Actual vs Predicted Deaths', 'Total Deaths')

# Bar chart for Total Injuries
plot_bar_chart(y_test_injuries.values, y_pred_injuries, 'Actual vs Predicted Injuries', 'Total Injuries')

# Plotting the evaluation results for each model
'''
# Bar chart for Accidents
plt.figure(figsize=(8, 6))
plt.bar(['MAE', 'MSE', 'R²'], [mae_accidents, mse_accidents, r2_accidents], color='skyblue')
plt.title("Accidents Model Evaluation")
plt.ylabel("Score")
plt.show()

# Bar chart for Deaths
plt.figure(figsize=(8, 6))
plt.bar(['MAE', 'MSE', 'R²'], [mae_deaths, mse_deaths, r2_deaths], color='salmon')
plt.title("Deaths Model Evaluation")
plt.ylabel("Score")
plt.show()

# Bar chart for Injuries
plt.figure(figsize=(8, 6))
plt.bar(['MAE', 'MSE', 'R²'], [mae_injuries, mse_injuries, r2_injuries], color='lightgreen')
plt.title("Injuries Model Evaluation")
plt.ylabel("Score")
plt.show()
'''

"""**Linear regression**"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_total_acc.csv')

# Prepare the features (X) and target variables (y)
X = data[['Death_and_injury_accidents', 'Property_damage_accidents', 'Death', 'Injured']]  # Features

# Separate target variables for accidents, deaths, and injuries
y_accidents = data['Total_Accidents']
y_deaths = data['Death']
y_injuries = data['Injured']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train_accidents, y_test_accidents = train_test_split(X, y_accidents, test_size=0.2, random_state=42)
_, _, y_train_deaths, y_test_deaths = train_test_split(X, y_deaths, test_size=0.2, random_state=42)
_, _, y_train_injuries, y_test_injuries = train_test_split(X, y_injuries, test_size=0.2, random_state=42)

# Initialize LinearRegression models for each target variable
model_accidents = LinearRegression()
model_deaths = LinearRegression()
model_injuries = LinearRegression()

# Train the models
model_accidents.fit(X_train, y_train_accidents)
model_deaths.fit(X_train, y_train_deaths)
model_injuries.fit(X_train, y_train_injuries)

# Make predictions for each model on the test set
y_pred_accidents = model_accidents.predict(X_test)
y_pred_deaths = model_deaths.predict(X_test)
y_pred_injuries = model_injuries.predict(X_test)

# Evaluate the models' performance
mae_accidents = mean_absolute_error(y_test_accidents, y_pred_accidents)
mse_accidents = mean_squared_error(y_test_accidents, y_pred_accidents)
r2_accidents = r2_score(y_test_accidents, y_pred_accidents)

mae_deaths = mean_absolute_error(y_test_deaths, y_pred_deaths)
mse_deaths = mean_squared_error(y_test_deaths, y_pred_deaths)
r2_deaths = r2_score(y_test_deaths, y_pred_deaths)

mae_injuries = mean_absolute_error(y_test_injuries, y_pred_injuries)
mse_injuries = mean_squared_error(y_test_injuries, y_pred_injuries)
r2_injuries = r2_score(y_test_injuries, y_pred_injuries)

# Print the evaluation metrics for each model
print("Accidents Model:")
print(f'MAE: {mae_accidents}, MSE: {mse_accidents}, R²: {r2_accidents}')

print("\nDeaths Model:")
print(f'MAE: {mae_deaths}, MSE: {mse_deaths}, R²: {r2_deaths}')

print("\nInjuries Model:")
print(f'MAE: {mae_injuries}, MSE: {mse_injuries}, R²: {r2_injuries}')

# Predictions for a specific test set (Optional)
print("\nPredictions for a specific test set (Accidents):")
print(y_pred_accidents[:10])  # Show first 10 predictions for accidents

print("\nPredictions for a specific test set (Deaths):")
print(y_pred_deaths[:10])  # Show first 10 predictions for deaths

print("\nPredictions for a specific test set (Injuries):")
print(y_pred_injuries[:10])  # Show first 10 predictions for injuries

# Plotting the predicted vs actual values for each model using line charts

# Line chart for Total Accidents
plt.figure(figsize=(7, 5))
plt.plot(y_test_accidents.values, label='Actual Accidents', color='blue')
plt.plot(y_pred_accidents, label='Predicted Accidents', color='red', linestyle='--')
plt.title('Actual vs Predicted Accidents', fontsize=10)
plt.xlabel('Data Points', fontsize=10)
plt.ylabel('Total Accidents', fontsize=10)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Line chart for Total Deaths
plt.figure(figsize=(7, 5))
plt.plot(y_test_deaths.values, label='Actual Deaths', color='blue')
plt.plot(y_pred_deaths, label='Predicted Deaths', color='red', linestyle='--')
plt.title('Actual vs Predicted Deaths', fontsize=10)
plt.xlabel('Data Points', fontsize=10)
plt.ylabel('Total Deaths', fontsize=10)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Line chart for Total Injuries
plt.figure(figsize=(7, 5))
plt.plot(y_test_injuries.values, label='Actual Injuries', color='blue')
plt.plot(y_pred_injuries, label='Predicted Injuries', color='red', linestyle='--')
plt.title('Actual vs Predicted Injuries', fontsize=10)
plt.xlabel('Data Points', fontsize=10)
plt.ylabel('Total Injuries', fontsize=10)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Function for plotting bar charts
def plot_bar_chart(actual, predicted, title, ylabel):
    indices = np.arange(len(actual))
    bar_width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(indices, actual, bar_width, label='Actual', color='blue')
    plt.bar(indices + bar_width, predicted, bar_width, label='Predicted', color='red', alpha=0.7)
    plt.title(title, fontsize=10)
    plt.xlabel('Data Points', fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.legend()
    #plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Bar chart for Total Accidents
plot_bar_chart(y_test_accidents.values, y_pred_accidents, 'Actual vs Predicted Accidents', 'Total Accidents')

# Bar chart for Total Deaths
plot_bar_chart(y_test_deaths.values, y_pred_deaths, 'Actual vs Predicted Deaths', 'Total Deaths')

# Bar chart for Total Injuries
plot_bar_chart(y_test_injuries.values, y_pred_injuries, 'Actual vs Predicted Injuries', 'Total Injuries')

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load data
data = {
    'City': ['Istanbul', 'Ankara', 'Izmir', 'Bursa', 'Konya'],
    'Latitude': [41.1082, 39.9208, 38.4192, 40.1443, 37.8716],
    'Longitude': [28.9784, 32.8541, 26.5187, 28.5550, 32.4847],
    '2021': [1887, 2289, 2642, 1348 , 1095],
    '2022': [1921, 2306, 2489, 1385, 1153],
    '2023': [2214, 2322, 2416, 1294, 1382],
    '2024': [1907, 2782 , 2670, 1253, 1438],
}

df = pd.DataFrame(data)

# Load Turkey shapefile (provide the path to the extracted shapefile)
turkey_map = gpd.read_file('/content/drive/My Drive/Colab Notebooks/Data_analytics/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')

# Filter for Turkey
turkey_map = turkey_map[turkey_map['NAME'] == 'Turkey']

# Plot map
fig, ax = plt.subplots(figsize=(12, 12))
turkey_map.plot(ax=ax, color='lightgray')

# Define colors for cities
colors = ['red', 'blue', 'green', 'black', 'purple']

# Plot data for each city
for i, row in df.iterrows():
    # Scatter point for city
    ax.scatter(row['Longitude'], row['Latitude'], color=colors[i], s=100, label=row['City'], alpha=0.8)

    # Annotate accident data for each year
    annotation = (
        f"2021: {row['2021']}\n"
        f"2022: {row['2022']}\n"
        f"2023: {row['2023']}\n"
        f"2024: {row['2024']}"
    )
    ax.text(
        row['Longitude'] + 0.3, row['Latitude'], annotation,
        fontsize=10, color=colors[i], ha='left', va='center'
    )

# Title and legend
plt.title('Total Accidents in Turkish Five Major Cities (2021-2024)', fontsize=10)
plt.xlabel('Longitude', fontsize=10)
plt.ylabel('Latitude', fontsize=110)
plt.legend(title='Cities', loc='upper right', fontsize=10)
plt.grid(False, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

import os

file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp'
print(os.path.exists(file_path))

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load Turkey's provincial shapefile
shapefile_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/gadm41_TUR_shp/gadm41_TUR_1.shp'
turkey_map = gpd.read_file(shapefile_path)

# Define the cities of interest, their provinces, and accident data
cities_data = {
    'Istanbul': {'province': 'Istanbul', 'accidents': [1887, 1921, 2214, 1907]},
    'Ankara': {'province': 'Ankara', 'accidents': [2289, 2306, 2322, 2782]},
    'Izmir': {'province': 'Izmir', 'accidents': [2642, 2489, 2416, 2670]},
    'Bursa': {'province': 'Bursa', 'accidents': [1348, 1385, 1294, 1253]},
    'Konya': {'province': 'Konya', 'accidents': [1095, 1153, 1382, 1438]}
}
city_colors = {
    'Istanbul': 'cyan',
    'Ankara': 'lightblue',
    'Izmir': 'lightgreen',
    'Bursa': 'yellow',
    'Konya': 'orange'
}

# Project to a planar CRS
turkey_map = turkey_map.to_crs(epsg=3857)

# Filter the shapefile for the cities of interest
filtered_map = turkey_map[turkey_map['NAME_1'].isin([v['province'] for v in cities_data.values()])]

# Plot map
fig, ax = plt.subplots(figsize=(12, 12))
turkey_map.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5)

# Plot each city's provincial boundary and add annotations for accidents
for city, data in cities_data.items():
    province_data = filtered_map[filtered_map['NAME_1'] == data['province']]
    province_data.plot(ax=ax, color=city_colors[city], label=f"{city} ({sum(data['accidents'])})")

    # Get the centroid of the province for annotation
    centroid = province_data.geometry.centroid
    for x, y in zip(centroid.x, centroid.y):
        annotation = (
            f"2021: {data['accidents'][0]}\n"
            f"2022: {data['accidents'][1]}\n"
            f"2023: {data['accidents'][2]}\n"
            f"2024: {data['accidents'][3]}"
        )
        ax.text(
            x, y, annotation, fontsize=11, fontweight='bold', color='black', ha='center', va='center'
        )

# Create a custom legend for city colors
legend_elements = [
    Patch(facecolor=color, edgecolor='black', label=city) for city, color in city_colors.items()
]
plt.legend(
    handles=legend_elements,
    title='Cities',
    loc='upper right',
    fontsize=10,
    title_fontsize=12
)

# Add title
plt.title('Total Accidents in Major Turkish Cities (2021-2024)',fontweight='bold', fontsize=11)

# Remove axis values and grid
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)

plt.tight_layout()
plt.show()

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load data
data = {
    'City': ['Istanbul', 'Ankara', 'Izmir', 'Bursa', 'Konya'],
    'Latitude': [41.0082, 39.9208, 38.4192, 40.1828, 37.8716],
    'Longitude': [28.9784, 32.8541, 27.1287, 29.0663, 32.4847],
    '2021': [12345, 6789, 4567, 3456, 2900],
    '2022': [13000, 7000, 4800, 3600, 3100],
    '2023': [14000, 7200, 5000, 3800, 3300],
    '2024': [15000, 7500, 5200, 4000, 3500],
}

df = pd.DataFrame(data)

# Load Turkey shapefile
turkey_map = gpd.read_file('/content/drive/My Drive/Colab Notebooks/Data_analytics/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')

# Filter for Turkey
turkey_map = turkey_map[turkey_map['NAME'] == 'Turkey']

# Convert the city data to a GeoDataFrame
geometry = gpd.points_from_xy(df['Longitude'], df['Latitude'])
geo_df = gpd.GeoDataFrame(df, geometry=geometry)

# Adjust Bursa and Istanbul positions slightly to avoid overlap
geo_df.loc[geo_df['City'] == 'Istanbul', 'Longitude'] += 0.2
geo_df.loc[geo_df['City'] == 'Bursa', 'Longitude'] -= 0.2

# Create buffers around city points for border visualization
geo_df['geometry'] = geo_df['geometry'].buffer(0.15)

# Plot map
fig, ax = plt.subplots(figsize=(12, 12))
turkey_map.plot(ax=ax, color='lightgray', edgecolor='black')

# Define colors for cities
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Plot city borders with color
for i, row in geo_df.iterrows():
    geo_df[geo_df['City'] == row['City']].plot(ax=ax, color=colors[i], alpha=0.6, label=row['City'])

    # Annotate accident data
    annotation = (
        f"2021: {row['2021']}\n"
        f"2022: {row['2022']}\n"
        f"2023: {row['2023']}\n"
        f"2024: {row['2024']}"
    )
    ax.text(
        row['geometry'].centroid.x + 0.3, row['geometry'].centroid.y,
        annotation, fontsize=10, color=colors[i], ha='left', va='center'
    )

# Title and legend
plt.title('Total Accidents in Turkish Cities (2021-2024)', fontsize=16)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.legend(title='Cities', loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()