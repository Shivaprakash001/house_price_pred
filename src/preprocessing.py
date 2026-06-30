import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Optional


class HousingPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for the Ames housing price prediction pipeline.

    This class encapsulates:
    1. Feature engineering: Combined FullBath and HalfBath into TotalBathrooms.
    2. Feature selection: Extracts GrLivArea, BedroomAbvGr, and TotalBathrooms.
    3. Imputation: Applies training-set medians to prevent data leakage.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None) -> None:
        """
        Initializes the preprocessor.

        Args:
            feature_cols: List of features to extract. Defaults to GrLivArea,
                         BedroomAbvGr, and TotalBathrooms.
        """
        self.feature_cols = feature_cols or [
            "GrLivArea",
            "BedroomAbvGr",
            "TotalBathrooms",
        ]
        self.medians_: Dict[str, float] = {}

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "HousingPreprocessor":
        """
        Fits the preprocessor by calculating medians on training features.

        Computing medians exclusively on the training split prevents data leakage.

        Args:
            X: Input training DataFrame.
            y: Ignored (included for scikit-learn API compatibility).

        Returns:
            The fitted instance of the preprocessor.
        """
        # Create a temporary copy to calculate intermediate features safely
        temp_df = X.copy()

        # Pre-fill individual bathroom fields with 0 for safety before calculation
        full_bath = temp_df["FullBath"].fillna(0)
        half_bath = temp_df["HalfBath"].fillna(0)
        temp_df["TotalBathrooms"] = full_bath + 0.5 * half_bath

        # Compute medians for target features
        features = temp_df[["GrLivArea", "BedroomAbvGr", "TotalBathrooms"]]
        for col in features.columns:
            self.medians_[col] = float(features[col].median())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by applying feature engineering and imputation.

        Does not modify the input DataFrame in-place.

        Args:
            X: Input DataFrame to transform.

        Returns:
            pd.DataFrame: A new DataFrame containing only the selected and processed features.
        """
        # Create copy to avoid modifying original dataframe
        df_copy = X.copy()

        # 1. Feature Engineering
        full_bath = df_copy["FullBath"].fillna(0)
        half_bath = df_copy["HalfBath"].fillna(0)
        df_copy["TotalBathrooms"] = full_bath + 0.5 * half_bath

        # 2. Select columns
        features = df_copy[["GrLivArea", "BedroomAbvGr", "TotalBathrooms"]].copy()

        # 3. Impute missing values using fitted training medians
        for col in features.columns:
            if col in self.medians_:
                features[col] = features[col].fillna(self.medians_[col])
            else:
                features[col] = features[col].fillna(0.0)

        return features
