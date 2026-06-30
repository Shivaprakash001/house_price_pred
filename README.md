# House Price Prediction using Multiple Linear Regression

This repository contains my submission for the Machine Learning Task as part of my internship at **SkillCraft Technology**. The objective of this project is to implement a Multiple Linear Regression model to predict residential house prices based on three foundational physical features: square footage, the number of bedrooms, and the number of bathrooms.

## 📌 Project Overview
Predicting property values is a standard regression problem in data science. This project leverages an Ordinary Least Squares (OLS) Multiple Linear Regression framework to analyze how structural space allocation impacts market valuation, working with Kaggle's *House Prices: Advanced Regression Techniques* dataset.

## 🛠️ Tech Stack & Dependencies
- **Language:** Python 3.13
- **IDE:** Visual Studio Code (VS Code)
- **Libraries used:**
  - `pandas` & `numpy` (For data loading, cleanup, and feature engineering)
  - `scikit-learn` (For dataset splitting, model training, and performance evaluation)

## 📊 Feature Selection & Engineering
The following core variables were extracted and processed from the raw datasets (`train.csv` and `test.csv`):
1. **Square Footage (`GrLivArea`):** Above-grade (ground) living area square feet.
2. **Bedrooms (`BedroomAbvGr`):** Total bedrooms above ground level.
3. **Bathrooms (`TotalBathrooms`):** A custom engineered feature combining full and half baths to represent total capacity, calculated mathematically as:
   TotalBathrooms = FullBath + (0.5 * HalfBath)

## 📈 Model Performance & Parameters
When executed, the pipeline split the training data (80% training, 20% validation) and yielded the following metrics and weights:

### Evaluation Metrics:
- **Root Mean Squared Error (RMSE):** \$53,371.56
- **R² Score (Variance Explained):** 0.6286
  *Insight: This simple 3-feature baseline model successfully explains approximately 62.86% of the variance in housing prices.*

### Learned Predictive Formula:
SalePrice = \$56,862.58 + (\$100.64 * GrLivArea) - (\$26,645.53 * BedroomAbvGr) + (\$27,083.21 * TotalBathrooms)

- **Intercept (Base Price):** \$56,862.58
- **Square Footage (`GrLivArea`):** Adds \$100.64 to the home value for every additional square foot.
- **Total Bathrooms:** Adds a strong premium of \$27,083.21 per bathroom unit.
- **Bedrooms (`BedroomAbvGr`):** Carries a negative weight (-\$26,645.53). This highlights a structural trade-off: adding more bedrooms within a fixed amount of square footage splits the home into smaller, more cramped rooms, which lowers overall market value in a strictly linear layout.

## 📁 Repository Structure
```text
├── predict_prices.py          # Core machine learning Python pipeline
├── submission_linear_reg.csv  # Final output predictions generated for the test set
└── README.md                  # Project documentation and internship report
