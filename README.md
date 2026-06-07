# bike-ride-forecasting
# Urban Mobility Demand Forecasting Engine 🗺️🚲

A supervised machine learning regression pipeline built to forecast hourly ride-sharing demand based on historical meteorological data, seasonal trends, and localized holiday features.

## 📈 Executive Summary
Predicting fleet allocation and balancing supply-demand ratios is critical for modern ride-hailing networks. This project implements an end-to-end data pipeline that sanitizes environmental data and evaluates multiple regression architectures to optimize forecasting accuracy.

## 🛠️ Tech Stack & Dependencies
- **Core Engine:** Python, Scikit-Learn
- **Data Engineering:** Pandas, NumPy, Python `holidays` library
- **Exploratory Data Analysis:** Seaborn, Matplotlib

## 🔬 Pipeline Architecture
1. **Feature Engineering:** Extracted granular temporal variables (hour, day, month, year) from raw timestamps. Integrated the native `holidays` package to dynamically flag country-specific statutory holiday parameters.
2. **Outlier Mitigation:** Engineered conditional thresholds to filter atmospheric anomalies (e.g., severe windspeed metrics (>32) and invalid humidity values), significantly reducing evaluation noise.
3. **Data Scaling:** Normalized continuous numerical variables utilizing Scikit-Learn's `StandardScaler` to optimize gradient descent metrics during linear modeling.
4. **Model Benchmarking:** Evaluated performance metrics across Linear Regression, Lasso, Ridge, and Random Forest Regressor models using Mean Absolute Error (MAE).

## 🏆 Final Evaluation Results
The models were validated using an unseen test split, evaluating the mean variation in absolute bike demand predictions:
- **Random Forest Regressor:** Selected as the production architecture due to superior performance in capturing non-linear feature interactions across complex micro-climate scenarios.
