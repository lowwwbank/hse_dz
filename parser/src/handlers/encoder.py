"""Encoder handler for converting categorical features to numeric."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.handlers.base import BaseHandler, DataContext


class EncoderHandler(BaseHandler):
    """Handler for encoding categorical features.

    Converts categorical columns (like city) to numeric representations
    using label encoding.
    """

    def __init__(self, categorical_columns: Optional[list[str]] = None) -> None:
        """Initialize the encoder handler.

        Args:
            categorical_columns: List of columns to encode. If None,
                will encode 'city_clean' by default.
        """
        super().__init__()
        self.categorical_columns = categorical_columns or ["city_clean"]
        self.encoders: dict[str, LabelEncoder] = {}

    def process(self, context: DataContext) -> DataContext:
        """Encode categorical features.

        Args:
            context: The data context containing the DataFrame.

        Returns:
            The context with encoded features.

        Raises:
            ValueError: If context.df is None.
        """
        if context.df is None:
            raise ValueError("No DataFrame to encode. Run previous handlers first.")

        context.log(self.name, "Encoding categorical features...")

        df = context.df.copy()

        for col in self.categorical_columns:
            if col not in df.columns:
                context.log(self.name, f"  Warning: Column '{col}' not found, skipping")
                continue

            context.log(self.name, f"  Encoding '{col}'...")

            # Fill missing values before encoding
            df[col] = df[col].fillna("unknown")

            encoder = LabelEncoder()
            encoded_col = f"{col}_encoded"
            df[encoded_col] = encoder.fit_transform(df[col].astype(str))

            self.encoders[col] = encoder

            # Update feature columns list
            if "feature_columns" in context.metadata:
                feature_cols = context.metadata["feature_columns"]
                if col in feature_cols:
                    # Replace original column with encoded version
                    idx = feature_cols.index(col)
                    feature_cols[idx] = encoded_col

            n_unique = len(encoder.classes_)
            context.log(self.name, f"    Encoded {n_unique} unique values")

        context.df = df
        context.metadata["encoders"] = self.encoders

        return context
