"""Salary regression model implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SalaryRegressionModel:
    """Regression model for salary prediction.

    Uses a pipeline with StandardScaler and GradientBoostingRegressor
    to predict salaries based on features extracted from HH.ru resume data.

    Internally applies log1p transform to target during training
    and expm1 during prediction for better performance.

    Features expected (in order):
        - gender: 1 (male), 0 (female), -1 (unknown)
        - age: Age in years
        - city_clean_encoded: Label-encoded city
        - experience_months: Total work experience in months
        - education_level: 0-4 ordinal scale
        - has_car_flag: 1 (has car), 0 (no car)

    Attributes:
        model: The underlying sklearn pipeline.
        is_fitted: Whether the model has been trained.
        feature_names: Names of expected features.
    """

    FEATURE_NAMES: list[str] = [
        "gender",
        "age",
        "city_clean_encoded",
        "experience_months",
        "education_level",
        "has_car_flag",
    ]

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.08,
        random_state: int = 42,
        use_log_transform: bool = True,
    ) -> None:
        """Initialize the salary regression model.

        Args:
            n_estimators: Number of boosting stages for GradientBoosting.
            max_depth: Maximum depth of individual trees.
            learning_rate: Learning rate for GradientBoosting.
            random_state: Random seed for reproducibility.
            use_log_transform: Whether to apply log1p/expm1 transform to target.
        """
        self._model: Pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                loss="huber",
                validation_fraction=0.1,
                n_iter_no_change=15,
                tol=1e-4,
                min_samples_split=5,
                min_samples_leaf=3,
                subsample=0.8,
            )),
        ])
        self._is_fitted: bool = False
        self._metadata: dict[str, Any] = {}
        self._use_log_transform: bool = use_log_transform

    @property
    def model(self) -> Pipeline:
        """Return the underlying sklearn pipeline."""
        return self._model

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been trained."""
        return self._is_fitted

    @property
    def feature_names(self) -> list[str]:
        """Return the names of expected features."""
        return self.FEATURE_NAMES.copy()

    @property
    def metadata(self) -> dict[str, Any]:
        """Return model metadata."""
        return self._metadata.copy()

    @property
    def use_log_transform(self) -> bool:
        """Return whether log transform is used."""
        return self._use_log_transform

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.float32],
    ) -> "SalaryRegressionModel":
        """Train the model on the given data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,) - salaries in rubles.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If X has wrong number of features.
        """
        self._validate_features(X)

        # Apply log transform to target for better distribution
        if self._use_log_transform:
            y_train = np.log1p(y)
        else:
            y_train = y

        self._model.fit(X, y_train)
        self._is_fitted = True

        self._metadata = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "feature_names": self.FEATURE_NAMES,
            "y_mean": float(np.mean(y)),
            "y_std": float(np.std(y)),
            "y_min": float(np.min(y)),
            "y_max": float(np.max(y)),
            "use_log_transform": self._use_log_transform,
        }

        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """Predict salaries for the given features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted salaries in rubles.

        Raises:
            ValueError: If model is not fitted or X has wrong shape.
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        self._validate_features(X)

        predictions = self._model.predict(X)

        # Inverse log transform
        if self._use_log_transform:
            predictions = np.expm1(predictions)

        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0.0)

        return predictions

    def _validate_features(self, X: NDArray[np.float32]) -> None:
        """Validate feature matrix dimensions.

        Args:
            X: Feature matrix to validate.

        Raises:
            ValueError: If X has wrong number of features.
        """
        expected_features = len(self.FEATURE_NAMES)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array.")
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {X.shape[1]}. "
                f"Expected features: {self.FEATURE_NAMES}"
            )

    def save(self, path: str | Path) -> None:
        """Save the model to a file.

        Args:
            path: Path to save the model.

        Raises:
            ValueError: If model is not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self._model,
            "is_fitted": self._is_fitted,
            "metadata": self._metadata,
            "feature_names": self.FEATURE_NAMES,
            "use_log_transform": self._use_log_transform,
        }

        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: str | Path) -> "SalaryRegressionModel":
        """Load a model from a file.

        Args:
            path: Path to the saved model.

        Returns:
            Loaded SalaryRegressionModel instance.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        instance = cls.__new__(cls)
        instance._model = model_data["model"]
        instance._is_fitted = model_data["is_fitted"]
        instance._metadata = model_data["metadata"]
        instance._use_log_transform = model_data.get("use_log_transform", False)

        return instance

    def get_feature_importances(self) -> dict[str, float]:
        """Get feature importances from the trained model.

        Returns:
            Dictionary mapping feature names to importance scores.

        Raises:
            ValueError: If model is not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted.")

        regressor = self._model.named_steps["regressor"]
        importances = regressor.feature_importances_

        return dict(zip(self.FEATURE_NAMES, importances))

    def score(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.float32],
    ) -> float:
        """Calculate R² score on the given data (in original scale).

        Args:
            X: Feature matrix.
            y: True target values in rubles.

        Returns:
            R² score.
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted.")

        self._validate_features(X)

        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)
