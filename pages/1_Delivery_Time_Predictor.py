import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
import catboost
import pickle
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm
import streamlit as st

st.set_page_config(page_title="Delivery Time Predictor", page_icon="🤖")
st.title("Delivery Time Predictor")

with open("8_final_data.csv", "rb") as f:
    df = pd.read_csv(f)
with open("models/Catboost.pkl","rb") as f:
    cat_pipe = pickle.load(f)
with open("models/Gradientboost.pkl","rb") as f:
    gb_pipe = pickle.load(f)
with open("models/Lightgbm.pkl","rb") as f:
    lgbm_pipe = pickle.load(f)
with open("models/Xgboost.pkl","rb") as f:
    xgb_pipe = pickle.load(f)

st.header("Enter Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Rider's Age")
with col2:
    rating = st.number_input("Rider's Rating")
col1, col2 = st.columns(2)
with col1:
    weather = st.selectbox("Weather Conditions",df['Weatherconditions'].unique())
with col2:
    traffic = st.selectbox("Traffic Conditions",df['traffic_type'].unique())
col1, col2 = st.columns(2)
with col1:
    vehicle = st.selectbox("Vehicle Condition",['Excellent','Good','Average','Bad'])
    if vehicle == 'Good':
        vehicle = 1
    elif vehicle == 'Excellent':
        vehicle = 0
    elif vehicle == 'Bad':
        vehicle = 3
    else:
        vehicle = 2
with col2:
    city = st.selectbox("City type",df['City_type'].unique())
col1, col2 = st.columns(2)
with col1:
    time = st.selectbox("Order Time of Day",df['order_time_of_day'].unique())
with col2:
    dist = st.number_input("Delivery Distance in Km")
col1, col2 = st.columns(2)
with col1:
    md = st.slider("No.of multiple deliveries",0,5)
with col2:
    hour = st.slider("Hour of Day",0,24)
if st.checkbox("Festival Day"):
    fest = 'yes'
else:
    fest = 'no'

data = [[age,rating,weather,traffic,vehicle,md,fest,city,hour,time,dist]]
cols = df.drop(columns=['time_taken']).columns.to_list()
test_df = pd.DataFrame(data, columns = cols)
if st.button("Predict"):
    y_xgb = xgb_pipe.predict(test_df)
    y_lgb = lgbm_pipe.predict(test_df)
    y_cat = cat_pipe.predict(test_df)
    y_gb = gb_pipe.predict(test_df)

    final_pred = (
            0.1 * y_xgb +
            0.4 * y_lgb +
            0.4 * y_cat +
            0.1 * y_gb
    )
    st.success("Your Food will be delivered in : {} minutes".format(round(final_pred[0])))

importances = cat_pipe.named_steps['model'].feature_importances_
features = df.drop('time_taken',axis=1).columns.to_list()

importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

fig,ax = plt.subplots(figsize=(10,8))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='Spectral',ax=ax)
ax.set_title("Feature Importance")
st.pyplot(fig)
