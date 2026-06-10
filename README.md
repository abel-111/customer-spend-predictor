# E-Commerce Customer Spending Prediction

## Overview

This project focuses on predicting customer spending in an e-commerce platform using machine learning. The dataset contains customer demographics, purchasing behavior, engagement metrics, session activity, and marketing preferences.

The goal is to build a regression model capable of estimating how much a customer is likely to spend based on historical behavior and customer characteristics.

---

## Project Objectives

* Predict customer spending (`total_spend_usd`)
* Perform complete data preprocessing and feature engineering
* Handle missing values and analyze outliers
* Compare multiple regression algorithms
* Identify the most important factors influencing customer spending
* Prepare the foundation for future CLV prediction and deployment

---

## Dataset Features

The dataset contains:

* Customer demographics
* Purchase history
* Session activity
* Marketing engagement
* Payment preferences
* Product category preferences
* Customer lifetime indicators

---

## Data Preprocessing

### Data Cleaning

Removed non-informative columns:

* customer_id
* name
* email

These columns do not contribute to customer spending prediction and may introduce unnecessary noise into the model.

### Feature Engineering

Date columns were converted into meaningful numerical features:

* customer_age_days
* days_since_first_order
* days_since_last_order
* days_since_first_session
* days_since_last_session

Original date columns were removed after feature extraction.

These features help capture customer recency, engagement, and purchasing history.

### Missing Value Handling

**Categorical Features**

* Mode Imputation

**Numerical Features**

* Median Imputation

Median was chosen for skewed numerical distributions to reduce the impact of extreme values.

### Outlier Analysis

Outliers were investigated using:

* Boxplots
* Histograms
* Descriptive Statistics

No outliers were removed because they represented genuine customer behavior such as:

* High-spending customers
* Loyal customers
* Highly active users

Removing these observations would reduce the model's ability to learn important business patterns.

### Target Transformation

The target variable `total_spend_usd` showed a strong positive skew.

A logarithmic transformation was applied:

```python
data['total_spend_log'] = np.log1p(data['total_spend_usd'])
```

This transformation helps improve model performance and stabilizes variance.

---

## Exploratory Data Analysis (EDA)

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
* Average Rating Given
* Total Sessions
* Customer Activity Metrics

The analysis was used to guide missing value treatment, outlier investigation, and feature engineering decisions.

---

## Categorical Encoding

### Label Encoding

Applied to:

* age_group
* preferred_device_ord
* preferred_source
* top_category_bought
* clv_tier

### One-Hot Encoding

Applied to:

* country
* preferred_payment
* preferred_device_sess
* preferred_source_sess

This converted categorical variables into machine-learning-friendly numerical representations.

---

## Model Development

### Train-Test Split

* 80% Training Data
* 20% Testing Data

### Feature Scaling

StandardScaler was applied before training Linear Regression models.

---

## Models Trained

### 1. Linear Regression

Used as the baseline model to establish initial performance.

### 2. Random Forest Regressor

Used to capture non-linear relationships between customer behavior and spending patterns.

### 3. Gradient Boosting Regressor

Implemented to improve predictive performance and analyze feature importance.

---

## Feature Leakage Detection

During model evaluation, unusually high performance was observed.

Investigation revealed that:

`avg_order_value = total_spend_usd / total_orders`

Since the feature directly contained information derived from the target variable, it created data leakage.

To ensure realistic model evaluation:

* avg_order_value was removed
* Models were retrained
* Performance was re-evaluated

This step significantly improved the reliability of the project.
total_orders was also removed because it dominated feature importance (0.932),
making all other features irrelevant to the model.

has_abandoned_cart showed high feature importance (0.879) but was intentionally
retained because it is a legitimate customer behavior feature, not derived from
the target variable. In real business, abandoned cart behavior is a strong
predictor of purchase intent.

---

## Feature Importance Analysis

Gradient Boosting Regressor was used to calculate feature importance scores.

The most influential features affecting customer spending were identified and visualized using bar charts.

This helps explain which customer behaviors contribute most to spending predictions.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Google Colab

---

## Project Status

### Completed

✅ Data Cleaning

✅ Feature Engineering

✅ Missing Value Handling

✅ Exploratory Data Analysis (EDA)

✅ Outlier Investigation

✅ Target Transformation

✅ Categorical Encoding

✅ Train-Test Split

✅ Feature Scaling

✅ Linear Regression Model

✅ Random Forest Regressor

✅ Gradient Boosting Regressor

✅ Feature Leakage Detection

✅ Feature Importance Analysis

✅Model Optimization

✅Hyperparameter Tuning

✅Cross Validation

### Currently Working On

* Model Optimization
* Hyperparameter Tuning
* Cross Validation

### Planned

* Customer Lifetime Value (CLV) Prediction
* Streamlit Web Application
* Interactive Dashboard
* Model Explainability (SHAP)
* Deployment

---

## Future Enhancements

### Customer Spending Prediction App

Develop a Streamlit-based web application where users can input customer information and receive spending predictions.

### Customer Lifetime Value Prediction

Extend the project to predict CLV tiers and long-term customer value.

### Explainable AI

Implement SHAP values and model interpretation techniques.

### Dashboard Integration

Build an interactive dashboard for business insights and customer analytics.

### Deployment

Deploy the final model and application using cloud platforms.

---

## Results

Successfully built and compared multiple machine learning regression models for customer spending prediction.

Key achievements:

* Converted raw business data into machine-learning-ready features.
* Detected and removed data leakage from avg_order_value and total_orders.
* Compared three regression models with the following results:

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.866 | 0.590 | 0.825 |
| Random Forest | 0.898 | 0.508 | 0.719 |
| Gradient Boosting | 0.902 | 0.501 | 0.705 |

* Gradient Boosting achieved the best performance with R² of 0.902.
* Identified the most important spending-related features.
* Saved the final model using Joblib for future deployment.

---

## Author

**Abel Shibu**

Aspiring Data Scientist | Machine Learning Enthusiast | Cybersecurity Learner

GitHub: abel-111

---

**Note:** This project is currently under active development. Future updates will include CLV prediction, model tuning, a Streamlit frontend, and deployment for real-world use.
