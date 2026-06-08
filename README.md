# E-Commerce Customer Spending Prediction

## Overview

This project focuses on analyzing customer purchasing behavior and building a machine learning model to predict customer spending in an e-commerce environment.

The project currently covers data cleaning, preprocessing, feature engineering, exploratory data analysis (EDA), outlier investigation, and preparation of a machine learning-ready dataset.

---

## Project Objective

Predict the total amount spent by a customer using demographic information, purchase history, customer engagement metrics, and behavioral data.

Target Variable:

* `total_spend_usd`

A logarithmic transformation is also applied to handle skewness:

* `total_spend_log`

---

## Dataset Features

The dataset contains customer-related information including:

* Demographics
* Purchase behavior
* Session activity
* Marketing engagement
* Payment preferences
* Product category preferences
* Customer lifetime indicators

---

## Data Preprocessing

### 1. Data Cleaning

Removed non-informative identifier columns:

* Customer ID
* Name
* Email

These columns do not contribute to customer spending prediction and may introduce noise into the model.

---

### 2. Feature Engineering

Raw date columns were transformed into meaningful numerical features:

Created Features:

* `customer_age_days`
* `days_since_first_order`
* `days_since_last_order`
* `days_since_first_session`
* `days_since_last_session`

Original date columns were removed after feature extraction.

---

### 3. Missing Value Handling

Missing values were handled using appropriate statistical techniques:

* Categorical features → Mode Imputation
* Numerical features → Median Imputation

This preserves the overall distribution while minimizing bias.

---

### 4. Exploratory Data Analysis (EDA)

Performed:

* Distribution Analysis
* Histograms
* Boxplots
* Summary Statistics

Features analyzed include:

* Age
* Total Orders
* Total Spend
* Average Order Value
* Average Discount Percentage
* Average Rating
* Total Sessions
* Customer Activity Metrics

---

### 5. Outlier Analysis

Potential outliers were identified using:

* Boxplots
* Descriptive Statistics
* IQR-based inspection

After investigation, no outliers were removed.

Reason:

The identified observations represented genuine customer behavior such as:

* High-value customers
* Loyal customers
* New customers
* Inactive customers

Removing these observations would reduce the model's ability to learn meaningful business patterns.

---

### 6. Target Transformation

The target variable exhibited strong positive skewness.

To reduce skewness and improve model performance:

```python
data['total_spend_log'] = np.log1p(data['total_spend_usd'])
```

This transformed target will be used during model training.

---

### 7. Encoding

Categorical features were converted into machine-learning-friendly formats using:

* Label Encoding
* One-Hot Encoding

Features encoded include:

* Country
* Age Group
* Payment Preferences
* Device Preferences
* Traffic Sources
* Product Categories

---

## Current Project Status

### Completed

* Dataset Cleaning
* Feature Engineering
* Missing Value Handling
* Exploratory Data Analysis
* Outlier Investigation
* Categorical Encoding
* Target Transformation
* Feature Selection

### In Progress

* Train-Test Split
* Feature Scaling
* Model Training
* Model Evaluation

### Planned

* Linear Regression
* Random Forest Regressor
* Model Comparison
* Hyperparameter Tuning
* Feature Importance Analysis
* Streamlit Web Application
* Model Deployment

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab

---

## Future Enhancements

* Customer Spending Prediction Web App
* Customer Lifetime Value (CLV) Prediction
* Advanced Feature Engineering
* Model Explainability
* Interactive Dashboard

---

## Author

Abel Shibu

Aspiring Data Scientist | Machine Learning Enthusiast | Cybersecurity Learner
