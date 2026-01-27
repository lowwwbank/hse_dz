"""Data splitter handler for creating X and y arrays."""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

from src.handlers.base import BaseHandler, DataContext


class DataSplitterHandler(BaseHandler):
    """Handler for splitting data into features (X) and target (y).

    This is typically the last handler in the chain. It creates
    numpy arrays and saves them to .npy files.
    """

    def __init__(
        self,
        remove_invalid: bool = True,
        normalize_target: bool = False,
    ) -> None:
        """Initialize the splitter handler.

        Args:
            remove_invalid: Whether to remove rows with invalid values.
            normalize_target: Whether to apply log transform to salary.
        """
        super().__init__()
        self.remove_invalid = remove_invalid
        self.normalize_target = normalize_target

    def _validate_and_filter(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> pd.DataFrame:
        """Validate and filter data before conversion.

        Args:
            df: DataFrame to validate.
            feature_cols: List of feature column names.
            target_col: Name of target column.

        Returns:
            Filtered DataFrame.
        """
        initial_count = len(df)

        # Remove rows with NaN in features or target
        all_cols = feature_cols + [target_col]
        df = df.dropna(subset=all_cols)

        # Remove rows with invalid age
        if "age" in feature_cols:
            df = df[(df["age"] > 0) & (df["age"] < 100)]

        # Remove rows with invalid salary
        df = df[(df[target_col] > 0) & (df[target_col] < 10_000_000)]

        # Remove rows with invalid experience
        if "experience_months" in feature_cols:
            df = df[(df["experience_months"] >= 0) & (df["experience_months"] < 600)]

        removed = initial_count - len(df)
        if removed > 0:
            print(f"  Removed {removed:,} invalid rows")

        return df

    def process(self, context: DataContext) -> DataContext:
        """Split data into X and y arrays and save to .npy files.

        Args:
            context: The data context containing the DataFrame.

        Returns:
            The context with X and y data and saved .npy files.

        Raises:
            ValueError: If context.df is None or required metadata is missing.
        """
        if context.df is None:
            raise ValueError("No DataFrame to split. Run previous handlers first.")

        if "feature_columns" not in context.metadata:
            raise ValueError("Feature columns not defined. Run FeatureExtractorHandler first.")

        if "target_column" not in context.metadata:
            raise ValueError("Target column not defined. Run FeatureExtractorHandler first.")

        context.log(self.name, "Creating X and y arrays...")

        df = context.df.copy()
        feature_cols = context.metadata["feature_columns"]
        target_col = context.metadata["target_column"]

        # Validate and filter data
        if self.remove_invalid:
            df = self._validate_and_filter(df, feature_cols, target_col)

        # Extract features and target
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)

        # Optionally normalize target (log transform for salary)
        if self.normalize_target:
            y = np.log1p(y)
            context.metadata["target_transformed"] = "log1p"

        context.x_data = X
        context.y_data = y

        context.log(self.name, f"X shape: {X.shape}")
        context.log(self.name, f"y shape: {y.shape}")

        # Save to .npy files
        x_path = os.path.join(context.output_dir, "x_data.npy")
        y_path = os.path.join(context.output_dir, "y_data.npy")

        np.save(x_path, X)
        np.save(y_path, y)

        context.log(self.name, f"Saved X to {x_path}")
        context.log(self.name, f"Saved y to {y_path}")

        context.metadata["x_path"] = x_path
        context.metadata["y_path"] = y_path
        context.metadata["final_shape"] = X.shape
        context.metadata["feature_names"] = feature_cols

        return context
