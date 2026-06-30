import sys
import os
import pickle
import numpy as np
import pandas as pd

# Add the parent directory of src to sys.path to enable importing src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import HousingPreprocessor


def run_prediction() -> None:
    """
    Executes the prediction pipeline:
    1. Loads the test dataset from data/test.csv.
    2. Loads the serialized model and preprocessor.
    3. Transforms the test set using the preprocessor (avoiding target leakage).
    4. Predicts house prices using the trained model.
    5. Clips negative predictions to zero.
    6. Saves the predictions in Kaggle-compatible format to outputs/submission.csv.
    """
    print("--- Starting Inference / Prediction Pipeline ---")

    # 1. Load Test Dataset
    test_path = os.path.join("data", "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    test_df = pd.read_csv(test_path)
    print(f"Test set loaded successfully. Shape: {test_df.shape}")

    # 2. Load trained preprocessor and model
    outputs_dir = "outputs"
    preprocessor_path = os.path.join(outputs_dir, "preprocessor.pkl")
    model_path = os.path.join(outputs_dir, "model.pkl")

    if not (os.path.exists(preprocessor_path) and os.path.exists(model_path)):
        raise FileNotFoundError(
            "Serialized model components missing. Please run src/train.py first."
        )

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 3. Preprocess Test Set
    X_test = preprocessor.transform(test_df)
    print("Test features engineered and imputed successfully using fitted medians.")

    # 4. Generate Predictions
    test_preds = model.predict(X_test)

    # 5. Phase 9: Clip negative predictions to zero
    test_preds_clipped = np.clip(test_preds, a_min=0, a_max=None)
    num_clipped = np.sum(test_preds != test_preds_clipped)
    if num_clipped > 0:
        print(f"  Warning: Clipped {num_clipped} negative price predictions to $0.00.")

    # 6. Generate Submission DataFrame
    submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": test_preds_clipped})

    # Save to outputs/submission.csv
    submission_path = os.path.join(outputs_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Final predictions successfully saved to: {submission_path}")
    print(submission.head())


if __name__ == "__main__":
    run_prediction()
