# Analysis and Prediction of Traffic Accidents in Major Cities of Türkiye (2021-2024)

![Python](https://img.shields.io/badge/Python-3.7%2B-blue) 
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-green) 
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.0%2B-orange) 
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.0%2B-yellow)

This project focuses on analyzing and predicting traffic accidents in five major cities of Türkiye (Istanbul, Ankara, Izmir, Bursa, and Konya) from **2021 to 2024**. The study uses machine learning models, including **Linear Regression (LR)** and **Random Forest Regressor (RFR)**, to predict accident trends, deaths, and injuries. The goal is to provide actionable insights for policymakers and urban planners to improve road safety and reduce traffic-related incidents.

---

## Features

- **Data Analysis**:
  - Exploratory Data Analysis (EDA) to identify trends, patterns, and key factors contributing to traffic accidents.
  - Visualization of accident hotspots, severity, and frequency across major cities in Türkiye.

- **Machine Learning Models**:
  - Predictive models to forecast accident severity, deaths, and injuries.
  - Comparison of **Linear Regression (LR)** and **Random Forest Regressor (RFR)** models using evaluation metrics such as **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R-squared (R²)**.

- **Geospatial Analysis**:
  - Mapping of accident-prone areas using geospatial tools.

- **Comprehensive Insights**:
  - Detailed analysis of monthly and yearly accident trends from 2021 to 2024.

---

## Requirements

To run this project, you need the following dependencies:

- Python 3.7 or higher
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Geopandas (for geospatial analysis)
- Jupyter Notebook (optional, for interactive analysis)

You can install the required libraries using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn geopandas jupyter
```

---

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/traffic-accidents-turkiye.git
   cd traffic-accidents-turkiye
   ```

2. **Run the Analysis**:
   - Open the Jupyter Notebook or Python script to perform data analysis and modeling:
     ```bash
     jupyter notebook analysis.ipynb
     ```

3. **Data Preprocessing**:
   - Clean and preprocess the dataset to handle missing values, outliers, and feature engineering.

4. **Model Training**:
   - Train machine learning models (Linear Regression and Random Forest Regressor) to predict accident trends, deaths, and injuries.

5. **Visualization**:
   - Generate visualizations such as heatmaps, bar charts, and geospatial maps to present insights.

---

## Dataset

The dataset includes traffic accident records from five major cities in Türkiye (2021-2024) with the following features:
- **Year**: 2021, 2022, 2023 and 2024.
- **Month**: Per month accidental data for the specified year.
- **Injured_Accident**: Number of accidents based on injury.
- **Death_Accident**:Number of accidents based on death.
- **Number_of_injured**: Number of injuries for each month.
- **Number_of_Death**: Number of death for each month.

---

## Project Structure

```
traffic-accidents-turkiye/
├── data/                    # Folder containing the dataset
│   ├── accidents_2021.csv
│   ├── accidents_2022.csv
│   ├── accidents_2023.csv
│   └── accidents_2024.csv
├── analysis.ipynb           # Jupyter Notebook for data analysis and visualization
├── models/                  # Trained machine learning models
├── README.md                # Project documentation
├── requirements.txt         # List of dependencies
└── results/                 # Folder to store output plots and reports
```

---

## Results

- **Accident Hotspots**:
  ![Hotspots](results/hotspots.png)

- **Severity Distribution**:
  ![Severity](results/severity_distribution.png)

- **Time-Series Trends**:
  ![Trends](results/accident_trends.png)

---

## Contributing

Contributions are welcome! If you'd like to improve this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- The dataset is sourced from the **Gendarmerie General Command (GGC)** of the Ministry of Internal Affairs of Türkiye.
- Special thanks to the open-source community for providing tools and libraries that made this project possible.

---

Let me know if you need further customization or additional sections! 🚀
