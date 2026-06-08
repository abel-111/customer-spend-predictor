# E-Commerce Customer Spending Prediction

## Overview

This project focuses on predicting customer spending behavior using machine learning techniques on an e-commerce dataset. The dataset contains customer demographics, purchasing behavior, session activity, engagement metrics, and customer lifetime value information.

The goal is to build a predictive model that estimates customer spending while performing comprehensive data preprocessing and feature engineering.

---

## Dataset Features

The dataset includes:

* Customer demographics (age, country, age group)
* Purchase history
* Session activity
* Payment preferences
* Marketing engagement
* Customer loyalty indicators
* Customer Lifetime Value (CLV) tier

---

## Data Preprocessing

### Data Cleaning

* Removed non-informative identifier columns:

  * Customer ID
  * Name
  * Email

### Missing Value Treatment

* Categorical features were imputed using mode values.
* Numerical features were imputed using median values after distribution analysis.

### Feature Engineering

Date-related columns were transformed into meaningful numerical features:

* Customer Age (days since signup)
* Days Since First Order
* Days Since Last Order
* Days Since First Session
* Days Since Last Session

Original date columns were removed after feature extraction.

### Outlier Analysis

Outliers were investigated using boxplots and the IQR method.

Features analyzed:

* Total Orders
* Total Spend
* Average Order Value
* Average Rating
* Total Sessions

The spending distribution was found to be highly right-skewed, indicating the presence of genuine high-value customers.

### Target Transformation

The target variable (`total_spend_usd`) showed significant positive skewness.

A logarithmic transformation was applied:

```python
data['total_spend_log'] = np.log1p(data['total_spend_usd'])
```

This helps reduce skewness and improve model performance.

### Encoding

* Label Encoding for ordinal and low-cardinality categorical features.
* One-Hot Encoding for nominal categorical features such as:

  * Preferred Payment Method
  * Preferred Device for Sessions
  * Preferred Session Source

---

## Machine Learning Objective

### Regression Task

Predict:

* Total Customer Spending (USD)

Target Variable:

* `total_spend_log`

Potential Models:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

---

## Current Status

✅ Data Cleaning Completed

✅ Missing Value Handling Completed

✅ Feature Engineering Completed

✅ Outlier Analysis Completed

✅ Target Transformation Completed

✅ Feature Encoding Completed

🔄 Model Training In Progress

🔄 Model Evaluation In Progress

🔄 Streamlit Deployment Planned

---

## Future Enhancements

* Train and compare multiple regression models
* Hyperparameter tuning
* Feature importance analysis
* Streamlit web application
* Model deployment
* Customer spending forecasting dashboard

---

## Author

Abel Shibu
Aspiring Data Scientist | Machine Learning Enthusiast | Cybersecurity Learner
