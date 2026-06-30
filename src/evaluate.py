import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Add parent directory of src to sys.path to enable importing src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import HousingPreprocessor


def load_assets():
    """Loads serialized training assets from the outputs directory."""
    outputs_dir = "outputs"
    model_path = os.path.join(outputs_dir, "model.pkl")
    preprocessor_path = os.path.join(outputs_dir, "preprocessor.pkl")
    split_data_path = os.path.join(outputs_dir, "split_data.pkl")

    if not (os.path.exists(model_path) and os.path.exists(preprocessor_path) and os.path.exists(split_data_path)):
        raise FileNotFoundError(
            "Serialized assets missing. Please run src/train.py first to train the model."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    with open(split_data_path, "rb") as f:
        split_data = pickle.load(f)

    return model, preprocessor, split_data


def evaluate_regression_model(y_true, y_pred):
    """Calculates standard regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def run_evaluation() -> None:
    """Executes the evaluation, interpretation, CV, and comparison stages."""
    print("--- Starting Model Evaluation Pipeline ---")

    # 1. Load Assets
    model, preprocessor, split_data = load_assets()
    X_train = split_data["X_train"]
    X_val = split_data["X_val"]
    y_train = split_data["y_train"]
    y_val = split_data["y_val"]
    X_raw_train = split_data["X_raw_train"]
    X_raw_val = split_data["X_raw_val"]

    # 2. Phase 6: Model Evaluation Metrics
    y_pred_val = model.predict(X_val)
    y_pred_train = model.predict(X_train)

    rmse_train, mae_train, r2_train = evaluate_regression_model(y_train, y_pred_train)
    rmse_val, mae_val, r2_val = evaluate_regression_model(y_val, y_pred_val)

    print("\n========================================")
    print("PHASE 6: MODEL EVALUATION METRICS (OLS)")
    print("========================================")
    print(f"Training Set Metrics:")
    print(f"  RMSE: ${rmse_train:,.2f}")
    print(f"  MAE:  ${mae_train:,.2f}")
    print(f"  R²:   {r2_train:.4f}")
    print(f"Validation Set Metrics:")
    print(f"  RMSE: ${rmse_val:,.2f}")
    print(f"  MAE:  ${mae_val:,.2f}")
    print(f"  R²:   {r2_val:.4f}")

    # Generate Phase 6 Plots
    sns.set_theme(style="whitegrid")
    
    # Plot A: Actual vs Predicted Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_val, y=y_pred_val, alpha=0.7, color="#1a73e8")
    # Draw reference diagonal line (y = x)
    min_val = min(y_val.min(), y_pred_val.min())
    max_val = max(y_val.max(), y_pred_val.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", lw=2, label="Perfect Predictor")
    plt.title("Actual vs Predicted House Prices (Validation Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Sale Price ($)", fontsize=11)
    plt.ylabel("Predicted Sale Price ($)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png", dpi=150)
    plt.close()

    # Plot B: Residual Plot
    residuals = y_val - y_pred_val
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred_val, y=residuals, alpha=0.7, color="#d93025")
    plt.axhline(y=0, color="black", linestyle="--", lw=2)
    plt.title("Residuals vs. Predicted Values (Validation Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Sale Price ($)", fontsize=11)
    plt.ylabel("Residuals ($) [Actual - Predicted]", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/residual_plot.png", dpi=150)
    plt.close()
    
    print("\nVisualizations saved: outputs/actual_vs_predicted.png and outputs/residual_plot.png")

    # 3. Phase 7: Model Interpretation
    print("\n========================================")
    print("PHASE 7: MODEL INTERPRETATION (OLS)")
    print("========================================")
    print(f"Intercept (Base Price): ${model.intercept_:,.2f}")
    features_list = X_train.columns
    coefficients = model.coef_
    
    for feat, coef in zip(features_list, coefficients):
        print(f"  {feat} Coefficient: {coef:,.2f}")
        
    print("\nInterpretation in Plain English:")
    print(f"1. Intercept: If a home has 0 sq ft of living area, 0 bedrooms, and 0 bathrooms, the model's base price estimate is ${model.intercept_:,.2f}.")
    print(f"2. GrLivArea: Keeping other features constant, adding 1 square foot of above-grade living area increases the estimated price by ${coefficients[0]:,.2f}.")
    print(f"3. BedroomAbvGr: Keeping other features constant, adding 1 bedroom above ground decreases the estimated price by ${abs(coefficients[1]):,.2f}. (This reflects splitting space into smaller, more cramped rooms).")
    print(f"4. TotalBathrooms: Keeping other features constant, adding 1 bathroom unit increases the estimated price by ${coefficients[2]:,.2f}.")

    # 4. Phase 8: Cross Validation (Leakage-free)
    print("\n========================================")
    print("PHASE 8: 5-FOLD CROSS VALIDATION (OLS)")
    print("========================================")
    
    # Load raw dataset for CV
    data_path = os.path.join("data", "train.csv")
    df_cv = pd.read_csv(data_path)
    X_cv_raw = df_cv.drop(columns=["SalePrice"])
    y_cv = df_cv["SalePrice"]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw), 1):
        # Split raw data
        X_fold_train_raw, X_fold_val_raw = X_cv_raw.iloc[train_idx], X_cv_raw.iloc[val_idx]
        y_fold_train, y_fold_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]
        
        # Fit Preprocessor on fold train, transform both splits
        fold_preprocessor = HousingPreprocessor()
        X_fold_train = fold_preprocessor.fit_transform(X_fold_train_raw)
        X_fold_val = fold_preprocessor.transform(X_fold_val_raw)
        
        # Train model
        fold_model = LinearRegression()
        fold_model.fit(X_fold_train, y_fold_train)
        
        # Predict & Evaluate
        y_fold_pred = fold_model.predict(X_fold_val)
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
        cv_rmse_scores.append(fold_rmse)
        print(f"  Fold {fold} RMSE: ${fold_rmse:,.2f}")
        
    mean_cv_rmse = np.mean(cv_rmse_scores)
    std_cv_rmse = np.std(cv_rmse_scores)
    print(f"\nAverage 5-Fold CV RMSE: ${mean_cv_rmse:,.2f} (± ${std_cv_rmse:,.2f})")

    # 5. Phase 10: Model Comparison
    print("\n========================================")
    print("PHASE 10: MODEL COMPARISON TABLE")
    print("========================================")
    
    # Train alternative models
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_val)
    rmse_dt, mae_dt, r2_dt = evaluate_regression_model(y_val, y_pred_dt)

    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_val)
    rmse_rf, mae_rf, r2_rf = evaluate_regression_model(y_val, y_pred_rf)

    # Compile results table
    results_data = {
        "Model": ["Linear Regression (OLS)", "Decision Tree Regressor", "Random Forest Regressor"],
        "RMSE ($)": [rmse_val, rmse_dt, rmse_rf],
        "MAE ($)": [mae_val, mae_dt, mae_rf],
        "R² Score": [r2_val, r2_dt, r2_rf]
    }
    
    df_comparison = pd.DataFrame(results_data)
    # Format for clean terminal printing
    print(df_comparison.to_markdown(index=False, floatfmt=(",.2f", ",.2f", ",.2f", ".4f")))
    
    # Save comparison to csv for reproducibility
    df_comparison.to_csv("outputs/model_comparison.csv", index=False)
    print("\nComparison table saved to: outputs/model_comparison.csv")
    
    # Diagnostics analysis on why tree models perform differently
    print("\nAnalysis & Insights:")
    print("- Why Random Forest out-performs OLS: Home prices exhibit non-linear interactions (e.g., adding a bathroom adds more value to larger square footage homes than smaller ones). Tree-based models capture these interactions automatically, whereas Linear Regression requires explicit feature crossing.")
    print("- Why Decision Tree might overfit: Single decision trees are prone to variance. They split until leaves are pure, memorizing training noise, which results in a lower validation R² than Random Forests.")


if __name__ == "__main__":
    run_evaluation()
