# 🚚 Delivery Time Prediction

## 🔍 Project Overview

The objective of this project is to **predict delivery time** using **real-world historical delivery data**. This end-to-end machine learning pipeline handles everything from data preprocessing to model deployment, ensuring a robust, interpretable, and high-performance solution.

---

## ⚙️ Workflow

### ✅ Data Preprocessing & Feature Engineering

- Cleaned and preprocessed delivery dataset with diverse features
- Handled:
  - **Missing values**
  - **Outliers**
  - **Categorical encodings**
  - **Feature extraction** such as:
    - Time of day
    - Delivery slot classification
    - Delivery sitance from location lat-longs

---

### 📊 Exploratory Data Analysis (EDA)

- Analyzed delivery patterns by:
  - Time
  - Ratings
  - Multiple Orders
  - Weather Conditions
  - Traffic conditions
- Used:
  - Kdeplots & boxplots
  - Correlation heatmaps
  - Trend analysis for delivery delays

---

### 🧠 Feature Selection

Applied multiple feature selection strategies to improve performance and interpretability:
- Tree-based model feature importances
- Permutation importance
- Correlation matrix-based reduction

---

### 🤖 Model Development & Selection

Trained and optimized the following models:
- Decision Tree
- Random Forest
- Gradient Boosting
- **XGBoost**
- **LightGBM**
- **CatBoost**

✅ **Hyperparameter tuning** using:
- GridSearchCV
- RandomizedSearchCV

### 🧪 Final Ensemble Model

Combined top-performing models using a **Weighted Voting Regressor**:
- Gradient Boost + XGBoost + LightGBM + CatBoost

📈 **Model Metrics**:
- R² Score: `0.819`
- MAE: `3.15 minutes`

---

## 🧩 Why It Matters

Accurate delivery time prediction has real-world applications in:

- 📦 Enhancing customer satisfaction  
- 🚚 Route and fleet optimization  
- 📈 Operational planning and logistics forecasting

---

## 💻 Tech Stack

- Python (Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, CatBoost)
- Streamlit (for interactive frontend)
- Matplotlib, Seaborn (for visualization)

