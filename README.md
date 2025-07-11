# 🏡 House Price Prediction using Machine Learning

Predicting house prices based on historical data is one of the most classic and practical applications of data science. This project applies various machine learning techniques to accurately predict the **sale price** of residential homes.

---

## 📌 Project Overview

This project uses a structured dataset containing house features to build a predictive model. The goal is to train regression models that can accurately estimate housing prices based on input features like square footage, number of bedrooms, location, etc.

---

## 📁 Dataset

- **Source**: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- **Target variable**: `SalePrice`
- **Features include**: 
  - LotArea
  - YearBuilt
  - OverallQual
  - TotalBsmtSF
  - GrLivArea
  - GarageCars
  - FullBath
  - Neighborhood (categorical)
  - and many more...

---

## 🛠️ Tools & Technologies

- **Programming Language**: Python
- **Libraries**:
  - `pandas`, `numpy` for data handling
  - `matplotlib`, `seaborn` for visualization
  - `scikit-learn` for preprocessing and ML models
  - `joblib` for model saving

---

## 📊 Project Pipeline

1. **Data Loading**
2. **Exploratory Data Analysis (EDA)**
3. **Missing Value Treatment**
4. **Feature Engineering**
5. **Encoding Categorical Features**
6. **Feature Scaling**
7. **Model Building (Linear Regression, Random Forest)**
8. **Model Evaluation**
9. **Prediction and Visualization**
10. **Model Saving for Deployment**

---

## 🧪 Models Used

| Model               | R² Score | RMSE     |
|--------------------|----------|----------|
| Linear Regression   | 0.83     | ~27,000  |
| Random Forest       | 0.89     | ~21,000  |

✅ Random Forest performed better and was selected for final predictions.

---

## 📈 Visualizations

- Correlation Heatmap
- Pairplot of top features
- Actual vs Predicted Prices Plot
- Distribution of Residuals

<p align="center">
  <img src="assets/price_plot.png" width="500"/>
</p>

---

## 📦 Folder Structure

# House-price-prediction
