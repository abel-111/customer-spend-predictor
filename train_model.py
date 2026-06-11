import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

data = pd.read_csv('ecommerce_data.csv')

data.drop(columns=['customer_id', 'name', 'email'], inplace=True)

date_cols = ['signup_date', 'first_order_date', 'last_order_date', 'first_session_date', 'last_session_date']
for col in date_cols:
    data[col] = pd.to_datetime(data[col], errors='coerce')

reference_date = max(data['last_order_date'].max(), data['last_session_date'].max())

data['customer_age_days']        = (reference_date - data['signup_date']).dt.days
data['days_since_first_order']   = (reference_date - data['first_order_date']).dt.days
data['days_since_last_order']    = (reference_date - data['last_order_date']).dt.days
data['days_since_first_session'] = (reference_date - data['first_session_date']).dt.days
data['days_since_last_session']  = (reference_date - data['last_session_date']).dt.days
data.drop(columns=date_cols, inplace=True)

cat_cols = ['country', 'age_group', 'preferred_payment', 'preferred_device_ord', 'preferred_source', 'top_category_bought', 'preferred_device_sess', 'preferred_source_sess']
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

num_cols = ['avg_rating_given', 'days_since_first_order', 'days_since_last_order', 'days_since_first_session', 'days_since_last_session']
for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

data['total_spend_log'] = np.log1p(data['total_spend_usd'])

le = LabelEncoder()
label_cols = ['age_group', 'preferred_device_ord', 'preferred_source', 'top_category_bought', 'clv_tier']
for col in label_cols:
    data[col] = le.fit_transform(data[col])

data = pd.get_dummies(data, columns=['country', 'preferred_payment', 'preferred_device_sess', 'preferred_source_sess'], drop_first=True, dtype=int)
data['marketing_opt_in'] = data['marketing_opt_in'].astype(int)

X = data.drop(columns=['total_spend_usd', 'total_spend_log', 'clv_tier', 'avg_order_value', 'total_orders'])
y = data['total_spend_log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'spending_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully")
print("Features used:", X.columns.tolist())