
# Access drive contents
"""

from google.colab import drive
drive.mount('/content/drive')

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/data_table/Konya_j_interp.csv'
data = pd.read_csv(file_path)

# Compute the total accidents column
#data['Total_Accidents'] = data['Death_and_injury_accidents'] + data['Property_damage_accidents']
# Compute the Total_Accidents column and insert it as the fifth column
data.insert(4, 'Total_Accidents', data['Death_and_injury_accidents'] + data['Property_damage_accidents'])

# Save the updated dataset
result_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_total_acc.csv'
data.to_csv(result_file_path, index=False)

# Ensure 'Month' is categorical and ordered correctly
data['Month'] = pd.Categorical(data['Month'], categories=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'], ordered=True)

# Plot Total Accidents for each month and year
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# Use lineplot for graph style
sns.lineplot(data=data, x='Month', y='Total_Accidents', hue='Year', marker='o', palette='tab10')

# Customize the plot
plt.title('Monthly Total Accidents in Konya (2021-2024)', fontsize=10)
plt.xlabel('Month', fontsize=10)
plt.ylabel('Total Accidents', fontsize=10)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Year', fontsize=10, title_fontsize=10)
plt.tight_layout()

# Save and display the plot
graph_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/Graphs/Konya/Konya_total_accidents.png'
plt.savefig(graph_file_path)
plt.show()

result_file_path, graph_file_path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/data_table/Konya_j_interp.csv'
data = pd.read_csv(file_path)

# Insert Total_Accidents as the fifth column
data.insert(4, 'Total_Accidents', data['Death_and_injury_accidents'] + data['Property_damage_accidents'])

# Save updated dataset
result_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_total_acc.csv'
data.to_csv(result_file_path, index=False)

# Ensure Month is categorical
data['Month'] = pd.Categorical(data['Month'], categories=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'], ordered=True)

# Define marker styles per year
marker_styles = {2021: 'o', 2022: 's', 2023: '^', 2024: 'D'}

# Create the plot
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# Plot each year manually for different markers
for year in sorted(data['Year'].unique()):
    subset = data[data['Year'] == year]
    sns.lineplot(data=subset, x='Month', y='Total_Accidents', label=str(year),
                 marker=marker_styles[year], linewidth=2, markersize=6)

# Customize fonts and style
plt.title('Monthly Total Accidents in Konya (2021–2024)', fontsize=12, color='black')
plt.xlabel('Month', fontsize=12, color='black')
plt.ylabel('Total Accidents', fontsize=12, color='black')
plt.xticks(rotation=45, fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.legend(title='Year', fontsize=12, title_fontsize=12, loc='upper left')
plt.tight_layout()

# Save and show the figure
graph_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/Graphs/new/Konya_total_accidents1.png'
plt.savefig(graph_file_path, dpi=300)
plt.show()

result_file_path, graph_file_path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_total_acc.csv'
data = pd.read_csv(file_path)

# Ensure that the 'Month' column is treated as a categorical variable with the correct order
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]
data['Month'] = pd.Categorical(data['Month'], categories=month_order, ordered=True)

# Aggregate total accidents by Year and Month
monthly_total_accidents = data.groupby(['Year', 'Month'])['Total_Accidents'].sum().unstack()

# Create pie charts for each year
for year in monthly_total_accidents.index:
    plt.figure(figsize=(6, 6))
    month_data = monthly_total_accidents.loc[year]

    # Plot the pie chart for the given year with better label positioning
    wedges, texts, autotexts = plt.pie(
        month_data,
        autopct=lambda p: f'{p * month_data.sum() / 100:.0f}',
        startangle=140,
        colors=sns.color_palette("Set3", len(month_data)),
        wedgeprops={'edgecolor': 'black'},
        pctdistance=0.85  # Moves the percentage text slightly inside the pie
    )

    # Adjust label positions to avoid overlap
    for text in texts:
        text.set(fontsize=10, ha='center', va='center')
    for autotext in autotexts:
        autotext.set(fontsize=10, ha='center', va='center')

    # Set title for the pie chart
    plt.title(f'Total Accidents Distribution by Month in Konya ({year})', fontsize=10)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Create legend with the corresponding month names
    plt.legend(wedges, month_data.index, title="Months", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # Adjust layout to prevent overlapping of elements
    plt.tight_layout()

    # Save the pie chart for each year
    pie_chart_file_path = f'/content/drive/My Drive/Colab Notebooks/Data_analytics/results/Graphs/Konya/Konya_total_accidents_pie_chart_{year}.png'
    plt.savefig(pie_chart_file_path)
    plt.show()

pie_chart_file_path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_total_acc.csv'
data = pd.read_csv(file_path)

# Ensure that the 'Month' column is treated as a categorical variable with the correct order
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]
data['Month'] = pd.Categorical(data['Month'], categories=month_order, ordered=True)

# Aggregate total accidents by month
monthly_total_accidents = data.groupby('Month')['Total_Accidents'].sum()

# Plot the pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    monthly_total_accidents,
    labels=monthly_total_accidents.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette("Set3", len(monthly_total_accidents))
)
plt.title('Total Accidents Distribution by Month in Konya (2021-2024)\n', fontsize=10)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Save the pie chart
pie_chart_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/Graphs/Konya/Konya_total_accidents_pie_chart.png'
plt.savefig(pie_chart_file_path)
plt.show()

pie_chart_file_path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_total_acc.csv'
data = pd.read_csv(file_path)

# Define the correct chronological order for months
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# Ensure that the 'Month' column is treated as a categorical variable with the correct order
data['Month'] = pd.Categorical(data['Month'], categories=month_order, ordered=True)

# Aggregate total accidents by Year and Month
monthly_total_accidents = data.groupby(['Year', 'Month'])['Total_Accidents'].sum().unstack()

# Create bar charts for each year
for year in monthly_total_accidents.index:
    plt.figure(figsize=(8, 6))
    month_data = monthly_total_accidents.loc[year]

    # Plot bar chart
    month_data.plot(kind='bar', color=sns.color_palette("Set3", len(month_data)), edgecolor='black')

    # Set title and labels
    plt.title(f'Total Accidents Distribution by Month in Konya ({year})', fontsize=10)
    plt.xlabel('Month', fontsize=10)
    plt.ylabel('Total Accidents', fontsize=10)

    # Add gridlines for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Set x-axis tick labels to be rotated for better readability
    plt.xticks(rotation=45, ha='right')

    # Save the bar chart for each year
    bar_chart_file_path = f'/content/drive/My Drive/Colab Notebooks/Data_analytics/results/Graphs/Konya/Konya_total_accidents_bar_chart_{year}.png'
    plt.tight_layout()  # Ensure everything fits without overlap
    plt.savefig(bar_chart_file_path)
    plt.show()

bar_chart_file_path

"""# Death graphs"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_total_acc.csv'
data = pd.read_csv(file_path)

# Define the correct chronological order for months
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# Ensure that the 'Month' column is treated as a categorical variable with the correct order
data['Month'] = pd.Categorical(data['Month'], categories=month_order, ordered=True)

# Aggregate total deaths by Year and Month
monthly_deaths = data.groupby(['Year', 'Month'])['Death'].sum().unstack()

# Create plots for each year
for year in monthly_deaths.index:
    # Pie chart for Death distribution by month
    plt.figure(figsize=(8, 8))
    death_data = monthly_deaths.loc[year]
    wedges, texts, autotexts = plt.pie(
        death_data,
        autopct=lambda p: f'{p * death_data.sum() / 100:.0f}',
        startangle=140,
        colors=sns.color_palette("Set3", len(death_data)),
        wedgeprops={'edgecolor': 'black'},
        pctdistance=0.85  # Moves the percentage text slightly inside the pie
    )
    for text in texts:
        text.set(fontsize=10, ha='center', va='center')
    for autotext in autotexts:
        autotext.set(fontsize=10, ha='center', va='center')

    plt.title(f'Deaths Distribution by Month in Konya ({year})', fontsize=10)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(wedges, death_data.index, title="Months", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    pie_chart_file_path = f'/content/drive/My Drive/Colab Notebooks/Data_analytics/results/Graphs/Konya/Konya_deaths_pie_chart_{year}.png'
    plt.savefig(pie_chart_file_path)
    plt.show()

    # Bar chart for Death distribution by month
    plt.figure(figsize=(8, 6))
    death_data.plot(kind='bar', color=sns.color_palette("Set3", len(death_data)), edgecolor='black')
    plt.title(f'Deaths Distribution by Month in Konya ({year})', fontsize=10)
    plt.xlabel('Month', fontsize=10)
    plt.ylabel('Total Deaths', fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    bar_chart_file_path = f'/content/drive/My Drive/Colab Notebooks/Data_analytics/results/Graphs/Konya/Konya_deaths_bar_chart_{year}.png'
    plt.tight_layout()
    plt.savefig(bar_chart_file_path)
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_total_acc.csv'
data = pd.read_csv(file_path)

# Define the correct chronological order for months
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# Ensure that the 'Month' column is treated as a categorical variable with the correct order
data['Month'] = pd.Categorical(data['Month'], categories=month_order, ordered=True)

# Aggregate total deaths by Year and Month
monthly_deaths = data.groupby(['Year', 'Month'])['Death'].sum().unstack()

# Define a color palette for the years
color_palette = sns.color_palette("Set1", len(monthly_deaths.index))

# Create a combined line plot for all years
plt.figure(figsize=(8, 6))

# Plot each year's data
for i, year in enumerate(monthly_deaths.index):
    # Plot the line
    plt.plot(monthly_deaths.columns, monthly_deaths.loc[year], marker='o', label=str(year),
             color=color_palette[i], linestyle='-', linewidth=2, markersize=8)

    # Annotate the values on each point of the line (month-wise)
    for j, month in enumerate(monthly_deaths.columns):
        plt.annotate(
            f'{monthly_deaths.loc[year][month]:.0f}',  # Death value (formatted as integer)
            (month, monthly_deaths.loc[year][month]),  # Position of the annotation
            textcoords="offset points",  # Move the text slightly off the point
            xytext=(0, 5),  # Offset by 5 points in y-direction
            ha='center',  # Horizontal alignment
            fontsize=10,  # Font size for the text
            color=color_palette[i]  # Set the text color to match the line
        )

# Set title and labels
plt.title('Deaths Trend by Month in Konya (All Years)', fontsize=10)
plt.xlabel('Month', fontsize=10)
plt.ylabel('Total Deaths', fontsize=10)

# Add gridlines for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Rotate x-ticks for readability
plt.xticks(rotation=45, ha='right')

# Add legend to differentiate the years
plt.legend(title="Year", loc="upper right", fontsize=8)

# Save and show the plot
line_plot_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/Graphs/Konya/Konya_deaths_combined_line_plot_with_values.png'
plt.tight_layout()
plt.savefig(line_plot_file_path)
plt.show()

line_plot_file_path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/data_table/Konya_j_interp.csv'
data = pd.read_csv(file_path)

# Compute the total accidents column
#data['Total_Accidents'] = data['Death_and_injury_accidents'] + data['Property_damage_accidents']
# Compute the Total_Accidents column and insert it as the fifth column
data.insert(4, 'Total_Accidents', data['Death_and_injury_accidents'] + data['Property_damage_accidents'])

# Save the updated dataset
result_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_total_acc.csv'
data.to_csv(result_file_path, index=False)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_total_acc.csv'
data = pd.read_csv(file_path)

# Define the correct chronological order for months
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# Ensure Month column is properly ordered
data['Month'] = pd.Categorical(data['Month'], categories=month_order, ordered=True)

# Aggregate total deaths by Year and Month
monthly_deaths = data.groupby(['Year', 'Month'])['Death'].sum().unstack()

# Define custom markers for each year
marker_styles = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
color_palette = sns.color_palette("tab10", len(monthly_deaths.index))

# Create the line plot
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# Plot each year with distinct marker and color
for i, year in enumerate(monthly_deaths.index):
    plt.plot(monthly_deaths.columns, monthly_deaths.loc[year],
             marker=marker_styles[i % len(marker_styles)],
             color=color_palette[i],
             label=str(year),
             linewidth=2,
             markersize=6)

    # Annotate each data point
    for j, month in enumerate(monthly_deaths.columns):
        plt.annotate(f'{monthly_deaths.loc[year][month]:.0f}',
                     (month, monthly_deaths.loc[year][month]),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center',
                     fontsize=12,
                     color='black')

# Set plot titles and labels
plt.title('Deaths Trend by Month in Konya (2021–2024)', fontsize=12, color='black')
plt.xlabel('Month', fontsize=12, color='black')
plt.ylabel('Total Deaths', fontsize=12, color='black')
plt.xticks(rotation=45, fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')

# Add grid and legend
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Year", loc="upper right", fontsize=12, title_fontsize=12)
plt.ylim(-0.3,8)
plt.tight_layout()

# Save and show the figure
line_plot_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/Graphs/new/Konya_deaths_combined_line_plot_with_values.png'
plt.savefig(line_plot_file_path, dpi=300)
plt.show()

line_plot_file_path

"""# injury Graph"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_total_acc.csv'
data = pd.read_csv(file_path)

# Define the correct chronological order for months
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# Ensure that the 'Month' column is treated as a categorical variable with the correct order
data['Month'] = pd.Categorical(data['Month'], categories=month_order, ordered=True)

# Aggregate total deaths by Year and Month
monthly_injured = data.groupby(['Year', 'Month'])['Injured'].sum().unstack()

# Define a color palette for the years
color_palette = sns.color_palette("Set1", len(monthly_injured.index))

# Create a combined line plot for all years
plt.figure(figsize=(8, 6))

# Plot each year's data
for i, year in enumerate(monthly_injured.index):
    # Plot the line
    plt.plot(monthly_injured.columns, monthly_injured.loc[year], marker='o', label=str(year),
             color=color_palette[i], linestyle='-', linewidth=2, markersize=8)

    # Annotate the values on each point of the line (month-wise)
    for j, month in enumerate(monthly_injured.columns):
        plt.annotate(
            f'{monthly_injured.loc[year][month]:.0f}',  # Death value (formatted as integer)
            (month, monthly_injured.loc[year][month]),  # Position of the annotation
            textcoords="offset points",  # Move the text slightly off the point
            xytext=(0, 5),  # Offset by 5 points in y-direction
            ha='center',  # Horizontal alignment
            fontsize=10,  # Font size for the text
            color=color_palette[i]  # Set the text color to match the line
        )

# Set title and labels
plt.title('Injured Trend by Month in Konya (2021-2024)', fontsize=10)
plt.xlabel('Month', fontsize=10)
plt.ylabel('Total injured', fontsize=10)

# Add gridlines for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Rotate x-ticks for readability
plt.xticks(rotation=45, ha='right')

# Add legend to differentiate the years
plt.legend(title="Year", loc="upper left")

# Save and show the plot
line_plot_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/Graphs/Konya/Konya_injured_combined_line_plot_with_values.png'
plt.tight_layout()
plt.savefig(line_plot_file_path)
plt.show()

line_plot_file_path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_total_acc.csv'
data = pd.read_csv(file_path)

# Define the correct chronological order for months
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# Ensure that the 'Month' column is treated as a categorical variable with the correct order
data['Month'] = pd.Categorical(data['Month'], categories=month_order, ordered=True)

# Aggregate total deaths by Year and Month
monthly_injured = data.groupby(['Year', 'Month'])['Injured'].sum().unstack()

# Define a color palette for the years
color_palette = sns.color_palette("Set1", len(monthly_injured.index))

# Create a combined line plot for all years
plt.figure(figsize=(8, 6))

# Plot each year's data
for i, year in enumerate(monthly_injured.index):
    # Plot the line
    plt.plot(monthly_injured.columns, monthly_injured.loc[year], marker='o', label=str(year),
             color=color_palette[i], linestyle='-', linewidth=2, markersize=8)

# Set title and labels
plt.title('Injured Trend by Month in Konya (2021-2024)', fontsize=10)
plt.xlabel('Month', fontsize=10)
plt.ylabel('Total injured', fontsize=10)

# Add gridlines for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Rotate x-ticks for readability
plt.xticks(rotation=45, ha='right')

# Add legend to differentiate the years
plt.legend(title="Year", loc="upper left")

# Save and show the plot
line_plot_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/Graphs/Konya/Konya_injured_combined_line_plot_no_values.png'
plt.tight_layout()
plt.savefig(line_plot_file_path)
plt.show()

line_plot_file_path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Bursa_j_total_acc.csv'
data = pd.read_csv(file_path)

# Define the correct chronological order for months
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# Ensure that the 'Month' column is treated as a categorical variable with the correct order
data['Month'] = pd.Categorical(data['Month'], categories=month_order, ordered=True)

# Aggregate total injuries by Year and Month
monthly_injured = data.groupby(['Year', 'Month'])['Injured'].sum().unstack()

# Define a color palette and marker styles for the years
color_palette = sns.color_palette("Set1", len(monthly_injured.index))
marker_styles = ['o', 's', 'D', 'v']  # circle, square, diamond, triangle

# Create a combined line plot for all years
plt.figure(figsize=(8, 6))

# Plot each year's data
for i, year in enumerate(monthly_injured.index):
    plt.plot(monthly_injured.columns, monthly_injured.loc[year],
             marker=marker_styles[i % len(marker_styles)],
             label=str(year),
             color=color_palette[i],
             linestyle='-', linewidth=2, markersize=8)
'''
    # Annotate the values on each point of the line (month-wise)
    for j, month in enumerate(monthly_injured.columns):
        plt.annotate(
            f'{monthly_injured.loc[year][month]:.0f}',
            (month, monthly_injured.loc[year][month]),
            textcoords="offset points",
            xytext=(0, 5),
            ha='center',
            fontsize=12,
            color='black'  # annotation text color
        )
'''
# Set title and labels
plt.title('Injured Trend by Month in Bursa (2021–2024)', fontsize=12, color='black')
plt.xlabel('Month', fontsize=12, color='black')
plt.ylabel('Total Injured', fontsize=12, color='black')

# Customize ticks and limits
plt.xticks(rotation=45, ha='right', fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
#plt.ylim(0, 80)  # You can adjust this range based on your data

# Add gridlines
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add legend
plt.legend(title="Year", loc="upper left", fontsize=12, title_fontsize=12)

# Save and show the plot
line_plot_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/Graphs/new/Bursa_injured_combined_line_plot_with_values.png'
plt.tight_layout()
plt.savefig(line_plot_file_path, dpi=300)
plt.show()

line_plot_file_path

"""# Total Calculations by year"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the existing dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Ankara_j_total_acc.csv'
data = pd.read_csv(file_path)

# Calculate the total accidents for each month (Death_and_injury_accidents + Property_damage_accidents)
data['Total_Accidents'] = data['Death_and_injury_accidents'] + data['Property_damage_accidents']

# Create an empty list to store the new rows (total values for each year)
new_rows = []

# Process each year individually and append the total row after December for each year
for year in data['Year'].unique():
    yearly_data = data[data['Year'] == year]

    # Calculate the total values for the year
    total_accidents = yearly_data['Total_Accidents'].sum()
    total_deaths = yearly_data['Death'].sum()
    total_injuries = yearly_data['Injured'].sum()

    # Create a new row for the total values (Year will be repeated for the new row)
    new_row = {
        'Year': year,
        'Month': 'Total',  # Indicate this is the total row for the year
        'Death_and_injury_accidents': total_accidents - yearly_data['Property_damage_accidents'].sum(),
        'Property_damage_accidents': total_accidents - yearly_data['Death_and_injury_accidents'].sum(),
        'Death': total_deaths,
        'Injured': total_injuries,
        'Total_Accidents': total_accidents
    }
    # Find the index after December of the respective year
    last_index = yearly_data.index[-1]

    # Insert the new row right after December
    data = pd.concat([data.iloc[:last_index+1], pd.DataFrame([new_row]), data.iloc[last_index+1:]], ignore_index=True)

# Save the new data with totals as a new CSV file
new_data_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Ankara_j_results_totals.csv'
data.to_csv(new_data_file_path, index=False)

# Display the modified table with totals
data

# Plot the total accidents, deaths, and injuries by year

# Plot Total Accidents by Year
plt.figure(figsize=(8, 6))
yearly_accidents = data[data['Month'] == 'Total'].groupby('Year')['Total_Accidents'].sum()
sns.barplot(x=yearly_accidents.index, y=yearly_accidents.values, palette='viridis')
plt.title('Total Accidents by Year', fontsize=10)
plt.xlabel('Year', fontsize=10)
plt.ylabel('Total Accidents', fontsize=10)
plt.tight_layout()
plt.show()

# Plot Total Deaths by Year
plt.figure(figsize=(8, 6))
yearly_deaths = data[data['Month'] == 'Total'].groupby('Year')['Death'].sum()
sns.barplot(x=yearly_deaths.index, y=yearly_deaths.values, palette='coolwarm')
plt.title('Total Deaths by Year', fontsize=10)
plt.xlabel('Year', fontsize=10)
plt.ylabel('Total Deaths', fontsize=10)
plt.tight_layout()
plt.show()

# Plot Total Injuries by Year
plt.figure(figsize=(8, 6))
yearly_injuries = data[data['Month'] == 'Total'].groupby('Year')['Injured'].sum()
sns.barplot(x=yearly_injuries.index, y=yearly_injuries.values, palette='magma')
plt.title('Total Injuries by Year', fontsize=10)
plt.xlabel('Year', fontsize=10)
plt.ylabel('Total Injuries', fontsize=10)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the existing dataset
file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_total_acc.csv'
data = pd.read_csv(file_path)

# Calculate the total accidents
data['Total_Accidents'] = data['Death_and_injury_accidents'] + data['Property_damage_accidents']

# Create a summary row for each year
new_rows = []
for year in data['Year'].unique():
    yearly_data = data[data['Year'] == year]

    total_accidents = yearly_data['Total_Accidents'].sum()
    total_deaths = yearly_data['Death'].sum()
    total_injuries = yearly_data['Injured'].sum()

    new_row = {
        'Year': year,
        'Month': 'Total',
        'Death_and_injury_accidents': total_accidents - yearly_data['Property_damage_accidents'].sum(),
        'Property_damage_accidents': total_accidents - yearly_data['Death_and_injury_accidents'].sum(),
        'Death': total_deaths,
        'Injured': total_injuries,
        'Total_Accidents': total_accidents
    }

    last_index = yearly_data.index[-1]
    data = pd.concat([data.iloc[:last_index+1], pd.DataFrame([new_row]), data.iloc[last_index+1:]], ignore_index=True)

# Save modified data
new_data_file_path = '/content/drive/My Drive/Colab Notebooks/Data_analytics/results/table/Konya_j_results_totals.csv'
data.to_csv(new_data_file_path, index=False)

# Set Seaborn theme and font size
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# ----- Plot 1: Total Accidents -----
plt.figure(figsize=(8, 6), dpi=300)
yearly_accidents = data[data['Month'] == 'Total'].groupby('Year')['Total_Accidents'].sum()
bar1 = sns.barplot(x=yearly_accidents.index, y=yearly_accidents.values, palette='viridis')

plt.title('Total Accidents by Year', fontsize=12, color='black')
plt.xlabel('Year', fontsize=12, color='black')
plt.ylabel('Total Accidents', fontsize=12, color='black')
# Removed value labels above bars
plt.tight_layout()
plt.show()

# ----- Plot 2: Total Deaths -----
plt.figure(figsize=(8, 6), dpi=300)
yearly_deaths = data[data['Month'] == 'Total'].groupby('Year')['Death'].sum()
bar2 = sns.barplot(x=yearly_deaths.index, y=yearly_deaths.values, palette='coolwarm')

plt.title('Total Deaths by Year', fontsize=12, color='black')
plt.xlabel('Year', fontsize=12, color='black')
plt.ylabel('Total Deaths', fontsize=12, color='black')
# Removed value labels above bars
plt.tight_layout()
plt.show()

# ----- Plot 3: Total Injuries -----
plt.figure(figsize=(8, 6), dpi=300)
yearly_injuries = data[data['Month'] == 'Total'].groupby('Year')['Injured'].sum()
bar3 = sns.barplot(x=yearly_injuries.index, y=yearly_injuries.values, palette='magma')

plt.title('Total Injuries by Year', fontsize=12, color='black')
plt.xlabel('Year', fontsize=12, color='black')
plt.ylabel('Total Injuries', fontsize=12, color='black')
# Removed value labels above bars
plt.tight_layout()
plt.show()
