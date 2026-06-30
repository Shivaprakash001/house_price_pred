import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load the Datasets (assuming they are in the root of your opened folder)
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# 2. Extract Features: Square Footage, Bedrooms, and calculated Bathrooms
def extract_features(df):
    full_bath = df["FullBath"].fillna(0)
    half_bath = df["HalfBath"].fillna(0)
    df["TotalBathrooms"] = full_bath + 0.5 * half_bath

    features = df[["GrLivArea", "BedroomAbvGr", "TotalBathrooms"]].copy()
    features = features.fillna(features.median())
    return features


X = extract_features(train_df)
y = train_df["SalePrice"]

# 3. Train/Validation Split (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the Model on Validation Data
y_pred_val = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2 = r2_score(y_val, y_pred_val)

print("--- Validation Metrics ---")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R² Score (Variance Explained): {r2:.4f}\n")

print("--- Model Parameters ---")
print(f"Intercept (Base Price): ${model.intercept_:,.2f}")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature} coefficient: {coef:,.2f}")

# 6. Predict on Test Set and Generate Kaggle Submission File
X_test = extract_features(test_df)
test_predictions = model.predict(X_test)
test_predictions = np.clip(test_predictions, a_min=0, a_max=None)

submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": test_predictions})
submission.to_csv("submission_linear_reg.csv", index=False)
print("\nSubmission file 'submission_linear_reg.csv' successfully created!")