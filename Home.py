import streamlit as st

st.set_page_config(page_title="About Project", page_icon="🚚")
st.markdown("""
# 🚚 Delivery Time Prediction

## 🔍 Project Overview

The main objective of this project is to **predict delivery time** using **historical delivery data**. The project follows a complete end-to-end machine learning workflow, including:

### ✅ Data Preprocessing & Feature Engineering
- Utilized a **real-world delivery dataset** with rich information across various delivery scenarios.
- Performed:
  - **Data cleaning**
  - **Missing value imputation**
  - **Feature extraction** (e.g., time of day, delivery slot)

### 📊 Exploratory Data Analysis (EDA)
- Identified key drivers of delivery delays using:
  - Distribution plots
  - Correlation heatmaps
  - Delay trend analysis
- Visualized delivery time behavior across Day Time, Weather Conditions, and traffic patterns.

### 📌 Feature Selection
- Applied multiple feature importance techniques:
  - Tree-based model feature importances
  - Permutation importance
  - Correlation-based reduction
- Selected the **most influential variables** to enhance model robustness.

### 🤖 Model Development & Selection
- Trained and evaluated a wide range of ML algorithms:
  - Decision Tree, Random Forest, Gradient Boost, **XGBoost**, **LightGBM**, **CatBoost**
- Performed **hyperparameter tuning** using GridSearchCV and RandomizedSearchCV.
- Created a **Weighted Ensemble Model** combining top-performing models:
  - Gradient Boost + XGBoost + LightGBM + CatBoost
- Optimized for both **R² score** and **Mean Absolute Error (MAE)** to ensure precision and generalization.
- The final model gives **R² score of 0.819** and **MAE of 3.15 min**.

---

## 🧩 Why It Matters?

In logistics and delivery operations, accurate delivery time prediction helps:
- 📦 Improve customer trust  
- 🚚 Optimize fleet management and routing  
- 📈 Drive operational efficiency and planning

---

## 👤 About the Developer

Developed by **[Ronit Balani]("https://www.linkedin.com/in/ronit-balani-845459258/")**, this app demonstrates the practical integration of machine learning and model interpretability.

""")
