import sys
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Add the parent directory of src to sys.path to enable importing src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import HousingPreprocessor


def run_training() -> None:
    """
    Executes the training pipeline:
    1. Loads the training dataset from data/train.csv.
    2. Performs the train/validation split (80/20) with random_state=42.
    3. Fits the preprocessor on training data and transforms training/validation features.
    4. Trains a Multiple Linear Regression model.
    5. Serializes the fitted preprocessor, trained model, and split datasets to outputs/.
    """
    print("--- Starting Model Training Pipeline ---")

    # 1. Load Dataset
    data_path = os.path.join("data", "train.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please check directory structure.")

    df = pd.read_csv(data_path)

    # Separate features and target
    X_raw = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    # 2. Phase 4: Data Splitting (Splitting raw data to prevent leakage)
    X_raw_train, X_raw_val, y_train, y_val = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )
    print(f"Data split successfully:")
    print(f"  Training Set Size: {X_raw_train.shape[0]} rows")
    print(f"  Validation Set Size: {X_raw_val.shape[0]} rows")

    # 3. Preprocessing (Fit on train only, transform train and validation)
    preprocessor = HousingPreprocessor()
    X_train = preprocessor.fit_transform(X_raw_train)
    X_val = preprocessor.transform(X_raw_val)
    print("Feature engineering and imputation applied successfully.")

    # 4. Phase 5: Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Multiple Linear Regression model trained successfully.")

    # Create outputs/ directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Serialize model and preprocessor for portability
    with open(os.path.join("outputs", "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join("outputs", "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    # Save processed splits for evaluate.py
    split_data = {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "X_raw_train": X_raw_train,
        "X_raw_val": X_raw_val,
    }
    with open(os.path.join("outputs", "split_data.pkl"), "wb") as f:
        pickle.dump(split_data, f)

    print("Model assets and split data successfully serialized to outputs/.")


if __name__ == "__main__":
    run_training()
