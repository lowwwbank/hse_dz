"""Predictor class for loading model and making predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.model import SalaryRegressionModel


class SalaryPredictor:
    """High-level interface for salary predictions.

    Handles model loading and provides a simple interface
    for making predictions from numpy arrays.

    Attributes:
        model: The loaded SalaryRegressionModel.
        model_path: Path to the loaded model file.
    """

    DEFAULT_MODEL_NAME: str = "salary_model.joblib"

    def __init__(self, model_path: str | Path | None = None) -> None:
        """Initialize the predictor with a model.

        Args:
            model_path: Path to the model file. If None, uses default
                        location in resources/ directory.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
        """
        if model_path is None:
            model_path = (
                Path(__file__).parent.parent / "resources" / self.DEFAULT_MODEL_NAME
            )

        self._model_path = Path(model_path)
        self._model = SalaryRegressionModel.load(self._model_path)

    @property
    def model(self) -> SalaryRegressionModel:
        """Return the underlying model."""
        return self._model

    @property
    def model_path(self) -> Path:
        """Return the path to the loaded model."""
        return self._model_path

    def predict(self, X: NDArray[np.float32]) -> list[float]:
        """Predict salaries for the given features.

        Args:
            X: Feature matrix of shape (n_samples, 6).
               Features: gender, age, city_encoded, experience_months,
                        education_level, has_car_flag

        Returns:
            List of predicted salaries in rubles.
        """
        predictions = self._model.predict(X)
        return predictions.tolist()

    def predict_from_file(self, file_path: str | Path) -> list[float]:
        """Load features from .npy file and predict salaries.

        Args:
            file_path: Path to the .npy file containing feature matrix.

        Returns:
            List of predicted salaries in rubles.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is invalid.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() != ".npy":
            raise ValueError(f"Expected .npy file, got: {file_path.suffix}")

        X = np.load(file_path)

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        return self.predict(X)

    def get_model_info(self) -> dict:
        """Get information about the loaded model.

        Returns:
            Dictionary with model metadata.
        """
        return {
            "model_path": str(self._model_path),
            "is_fitted": self._model.is_fitted,
            "feature_names": self._model.feature_names,
            "metadata": self._model.metadata,
        }
