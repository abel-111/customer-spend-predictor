# Customer Spend Predictor

## Overview

Customer Spend Predictor is a machine learning project focused on preparing e-commerce customer data for predictive analytics. The project currently includes data cleaning, missing value treatment, outlier handling, and feature encoding to build a high-quality dataset for future customer spending prediction models.

**Project Status:** 🚧 In Development

## Features Completed

### Data Loading and Inspection

* Loaded the e-commerce customer dataset using Pandas.
* Explored dataset structure, data types, and missing values.
* Analyzed dataset dimensions and feature information.

### Missing Value Treatment

Handled missing values using appropriate statistical methods:

* Mode imputation for categorical features.
* Median imputation for numerical features with skewed distributions.
* Verified successful removal of missing values.

### Data Visualization

Performed visual analysis using:

* Histograms to understand data distributions.
* Boxplots to identify outliers in numerical features.

### Outlier Detection and Removal

Applied the Interquartile Range (IQR) method to detect and remove outliers from:

* Total Orders
* Total Spend (USD)
* Average Order Value
* Average Rating Given
* Total Sessions

### Feature Encoding

Converted categorical variables into machine-learning-friendly formats.

#### Label Encoding

Applied to:

* Name
* Email
* Country
* Age Group
* Preferred Device (Ordered)
* Preferred Source
* Top Category Bought
* CLV Tier
* Signup Date
* First Order Date
* Last Order Date
* First Session Date
* Last Session Date

#### Binary Encoding

Applied to:

* Marketing Opt-In (True/False → 1/0)

#### One-Hot Encoding

Applied to:

* Preferred Payment Method
* Preferred Device Session
* Preferred Source Session

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Current Progress

* [x] Data Loading
* [x] Data Inspection
* [x] Missing Value Handling
* [x] Data Visualization
* [x] Outlier Detection & Removal
* [x] Feature Encoding
* [ ] Exploratory Data Analysis (EDA)
* [ ] Feature Engineering
* [ ] Model Training
* [ ] Total Spend Prediction
* [ ] Frontend Development
* [ ] Deployment

## Future Goals

* Perform detailed exploratory data analysis.
* Train regression models to predict customer spending.
* Compare multiple machine learning algorithms.
* Build an interactive frontend for predictions.
* Deploy the application for public use.

## Project Structure

```text
Customer-Spend-Predictor/
│
├── data/
├── notebooks/
├── project1.py
├── README.md
├── requirements.txt
└── models/
```

## Author

**Abel Shibu**

Aspiring Data Analyst | Machine Learning Enthusiast | Cybersecurity Learner
