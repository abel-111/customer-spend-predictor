import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('spending_model.pkl')

st.title("E-Commerce Customer Spending Prediction")
st.write("Enter customer details to predict total spending.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
has_abandoned_cart = st.selectbox("Has Abandoned Cart?", [0, 1])
is_repeat_customer = st.selectbox("Is Repeat Customer?", [0, 1])
marketing_opt_in = st.selectbox("Marketing Opt In?", [0, 1])
total_sessions = st.number_input("Total Sessions", min_value=1, max_value=500, value=10)
avg_discount_pct = st.number_input("Average Discount %", min_value=0.0, max_value=100.0, value=10.0)
avg_rating_given = st.number_input("Average Rating Given", min_value=1.0, max_value=5.0, value=3.0)
customer_age_days = st.number_input("Customer Age (Days)", min_value=1, max_value=3000, value=365)
days_since_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=1000, value=30)
country = st.selectbox("Country", ['IN', 'US', 'GB', 'AU', 'BR', 'CA', 'DE', 'ES', 'FR', 'JP', 'MX', 'NL', 'PL', 'SE', 'SG', 'ZA'])

if st.button("Predict Spending"):
    input_dict = {
        'age': age,
        'age_group': 1,
        'marketing_opt_in': marketing_opt_in,
        'avg_discount_pct': avg_discount_pct,
        'preferred_device_ord': 0,
        'preferred_source': 0,
        'top_category_bought': 0,
        'avg_rating_given': avg_rating_given,
        'total_sessions': total_sessions,
        'has_abandoned_cart': has_abandoned_cart,
        'is_repeat_customer': is_repeat_customer,
        'customer_age_days': customer_age_days,
        'days_since_first_order': customer_age_days,
        'days_since_last_order': days_since_last_order,
        'days_since_first_session': customer_age_days,
        'days_since_last_session': days_since_last_order,
        'country_AU': 1 if country == 'AU' else 0,
        'country_BR': 1 if country == 'BR' else 0,
        'country_CA': 1 if country == 'CA' else 0,
        'country_DE': 1 if country == 'DE' else 0,
        'country_ES': 1 if country == 'ES' else 0,
        'country_FR': 1 if country == 'FR' else 0,
        'country_GB': 1 if country == 'GB' else 0,
        'country_IN': 1 if country == 'IN' else 0,
        'country_JP': 1 if country == 'JP' else 0,
        'country_MX': 1 if country == 'MX' else 0,
        'country_NL': 1 if country == 'NL' else 0,
        'country_PL': 1 if country == 'PL' else 0,
        'country_SE': 1 if country == 'SE' else 0,
        'country_SG': 1 if country == 'SG' else 0,
        'country_US': 1 if country == 'US' else 0,
        'country_ZA': 1 if country == 'ZA' else 0,
        'preferred_payment_cod': 0,
        'preferred_payment_paypal': 0,
        'preferred_payment_wallet': 0,
        'preferred_device_sess_mobile': 0,
        'preferred_device_sess_tablet': 0,
        'preferred_source_sess_email': 0,
        'preferred_source_sess_organic': 0,
        'preferred_source_sess_paid': 0,
        'preferred_source_sess_referral': 0,
        'preferred_source_sess_social': 0
    }

    input_df = pd.DataFrame([input_dict])
    prediction_log = model.predict(input_df)[0]
    prediction_usd = max(0, np.expm1(prediction_log))
    st.success(f"Predicted Customer Spending: ${prediction_usd:.2f}")