
# Access drive contents
"""

from google.colab import drive
drive.mount('/content/drive')

"""# data interpolation"""

import pandas as pd

# Load the data from the CSV file
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/data_table/konya_j_trimmed.csv'

try:
    data = pd.read_csv(file_path)

    # Ensure the column is numeric
    data['Death_and_injury_accidents'] = pd.to_numeric(data['Death_and_injury_accidents'], errors='coerce')

    # Handle columns with entirely missing values
    if data['Death_and_injury_accidents'].isna().all():
        data['Death_and_injury_accidents'] = data['Death_and_injury_accidents'].fillna(method='ffill').fillna(method='bfill')

    # Interpolate the missing data (linear interpolation) and round to integers
    data_interpolated = data.interpolate(method='linear', axis=0)
    data_interpolated = data_interpolated.round(0).astype(int, errors='ignore')

    # Save the new interpolated data to a new CSV file
    new_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/data_table/Konya_j_interp.csv'
    data_interpolated.to_csv(new_file_path, index=False)

    new_file_path
except FileNotFoundError:
    "File not found. Please check the file path and try again."
import pandas as pd

# Load the CSV file
csv_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/data_table/Izmir_j_interp.csv'
dataset = pd.read_csv(csv_path)

# Calculate the shape
rows, columns = dataset.shape

# Display the shape
print(f"Dataset shape: {rows} rows, {columns} columns")