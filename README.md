# 🏡 House Price Prediction Pipeline: Production-Grade ML Repository

This repository implements a modular, production-grade machine learning pipeline for predicting residential housing prices based on Kaggle's **House Prices: Advanced Regression Techniques** dataset. 

Developed with a professional software engineering layout, this project features reusable modules, an exploratory notebook, validation visualizer outputs, leakage-free cross-validation, and comparative analysis with tree-based regressors.

---

## 📌 Project Structure

The project conforms to the standard industry layout for data science repositories:

```text
HOUSEPRICEPREDICTION-main/
│
├── data/
│     train.csv               # Raw Kaggle training dataset (read-only)
│     test.csv                # Raw Kaggle test dataset (read-only)
│
├── notebooks/
│     01_EDA.ipynb            # Jupyter notebook documenting exploratory data analysis
│
├── src/
│     __init__.py             # Makes src a Python package
│     preprocessing.py        # Custom preprocessor with leakage-free imputation & feature engineering
│     train.py                # Runner script to split data, fit pipelines, and save models
│     evaluate.py             # Computes metrics, plots residuals, runs CV, and compares models
│     predict.py              # Feeds test.csv through preprocessor and generates submission.csv
│
├── outputs/
│     actual_vs_predicted.png # Scatter plot comparing true prices vs model estimations
│     residual_plot.png       # Diagnostic plot mapping model errors across price levels
│     model_comparison.csv    # Comparative data frame for regression models
│     submission.csv          # Final Kaggle-compatible CSV predictions
│     model.pkl               # Serialized OLS model binary
│     preprocessor.pkl        # Serialized preprocessor binary
│
├── README.md                 # Complete project documentation and walkthrough
└── requirements.txt          # Python dependencies tracking file
```

### Why is this structure useful?
1. **Separation of Concerns:** Separates data exploration (`notebooks/`), core ETL and feature processing (`preprocessing.py`), model operations (`train.py`, `predict.py`), and analytics (`evaluate.py`).
2. **Reproducibility:** Code execution is standardized. Medians and parameters are fitted solely on the training split, preventing **data leakage**.
3. **Traceability:** Outputs and figures are generated programmatically and saved to `outputs/` rather than bloating the main source tree.

---

## 🚀 Installation & Usage

### 1. Requirements Setup
This codebase requires Python 3.12+ and uses a virtual environment for library tracking.
```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Execution Pipeline
Run the modular pipeline stages in order from the repository root:
```bash
# Step 1: Run Exploratory Data Analysis & generate plots
python3 notebooks/generate_eda.py

# Step 2: Split data, run preprocessing, and train the model
python3 src/train.py

# Step 3: Run model diagnostics, cross-validation, and model comparisons
python3 src/evaluate.py

# Step 4: Run inference on the test set and output Kaggle predictions
python3 src/predict.py
```

---

## 📊 Exploratory Data Analysis (EDA) Summary

Our EDA revealed critical structural behaviors in the dataset:
* **Target Distribution Skewness:** The `SalePrice` exhibits positive (right) skewness with a median of \$163,000 and a mean of \$180,921. This confirms that luxury homes stretch the regression line's tail.
* **Feature Interdependence:** `GrLivArea` (above-ground square footage) has a strong correlation ($r = 0.709$) with price, and `TotalBathrooms` correlates strongly ($r = 0.631$). Standalone bedroom counts show a weak correlation ($r = 0.168$) due to structural trade-offs.
* **Extreme Outliers:** In the scatter plot of `GrLivArea` vs `SalePrice`, we identified two high-leverage outliers: homes with living areas $>4,000$ sq ft that sold for under \$200,000. Under OLS optimization, these outliers draw the line of best fit down, introducing prediction bias.

---

## 📐 Feature Engineering & Preprocessing

To represent residential utility accurately, we engineered a single metric:
$$\text{TotalBathrooms} = \text{FullBath} + 0.5 \times \text{HalfBath}$$
*Rationale:* Half baths add utility but lack tub/shower installations. Combining them provides a single, high-fidelity capacity score.

### 🛡️ Leakage-Free Preprocessing
We created a custom `HousingPreprocessor` that inherits from scikit-learn's `BaseEstimator` and `TransformerMixin`. To prevent target leakage:
1. `fit()` computes median values exclusively on the training subset.
2. `transform()` applies these medians to missing values in validation and test splits, guaranteeing that test metrics are not polluted by validation-set properties.

---

## 📉 Model Performance & Diagnostic Results

### 1. Ordinary Least Squares (OLS) Regression
We split the training data (80% training, 20% validation) with a fixed seed (`random_state=42`). 

#### Model Formula Learned:
$$\text{SalePrice} = \$56,862.58 + (\$100.64 \times \text{GrLivArea}) - (\$26,645.53 \times \text{BedroomAbvGr}) + (\$27,083.21 \times \text{TotalBathrooms})$$

#### Coefficient Interpretation:
* **Intercept (\$56,862.58):** The baseline price estimate of a home with zero square feet, bedrooms, and bathrooms.
* **Living Area (`GrLivArea` Coefficient: +\$100.64):** Keeping other features constant, every additional square foot of living area adds \$100.64 to the home's value.
* **Total Bathrooms (`TotalBathrooms` Coefficient: +\$27,083.21):** Adding a bathroom unit adds a major premium of \$27.1k.
* **Bedrooms (`BedroomAbvGr` Coefficient: -\$26,645.53):** Keeping square footage constant, adding an extra bedroom decreases estimated value by \$26.6k. This reflects the *Bedroom Coefficient Paradox*: dividing a fixed amount of space into more rooms creates smaller, cramped rooms, reducing market value.

### 2. Validation Metrics (OLS)
* **Root Mean Squared Error (RMSE):** **\$53,371.56**
* **Mean Absolute Error (MAE):** **\$36,569.64**
* **R² Score (Variance Explained):** **0.6286** (Our simple 3-feature baseline explains ~62.9% of the price variance).

### 3. 5-Fold Cross-Validation
To obtain a more stable generalization score, we performed 5-Fold Cross Validation:
* **Average CV RMSE:** **\$52,093.49**
* **Standard Deviation:** **\$6,951.66**
* *Why use CV?* A single validation split is subject to variance; if our split accidentally holds only average-priced homes, the metrics will look misleadingly good. Cross-validation divides the data 5 times, training 5 models, and averaging performance to give a realistic range of real-world error.

---

## 🏆 Model Comparison Table

We compared our baseline OLS model with non-linear tree algorithms:

| Model | RMSE ($) | MAE ($) | R² Score |
|:---|---:|---:|---:|
| **Linear Regression (OLS)** | 53,371.56 | 36,569.64 | 0.6286 |
| **Decision Tree Regressor** | 50,645.14 | 35,135.66 | 0.6656 |
| **Random Forest Regressor** | **43,235.32** | **29,877.00** | **0.7563** |

### Why do tree models perform better?
* **Non-linear Interactions:** A bathroom in a 5,000 sq ft mansion adds more value than a bathroom in a 500 sq ft cottage. Linear models cannot capture this synergy unless we engineer explicit interaction terms (e.g., $\text{GrLivArea} \times \text{TotalBathrooms}$).
* **Ensemble Learning:** Random Forest builds 100 decision trees, each trained on bootstrapped subsets of training data and features. It averages their predictions, reducing the variance and overfitting associated with single decision trees.

---

## 📈 Future Improvements Roadmap

To push the accuracy of this pipeline further:
1. **Target Log-Transform:** Apply $y_{new} = \log(y)$ to normalize the target distribution and satisfy homoscedasticity.
2. **Outlier Filtering:** Remove the two extreme outliers identified in the EDA phase.
3. **Feature Expansion:** Introduce quality ratings (`OverallQual`, `KitchenQual`) and neighborhood location codes.
4. **Regularization:** Implement Ridge ($L_2$) or Lasso ($L_1$) regression to penalize large weights and perform automatic feature selection.
