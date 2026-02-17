"""Level classifier for IT developers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LevelClassifier:
    """Classifier for predicting IT developer level.

    Predicts junior/middle/senior level based on resume features.

    Features expected:
        - gender: 1 (male), 0 (female), -1 (unknown)
        - age: Age in years
        - salary: Expected salary in rubles
        - city_encoded: Label-encoded city
        - experience_months: Total work experience in months
        - is_full_time: 1 (full-time), 0 (other)
        - education_level: 0-4 ordinal scale
        - has_car: 1 (has car), 0 (no car)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        class_weight: str = "balanced",
        random_state: int = 42,
    ) -> None:
        """Initialize the classifier.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees.
            class_weight: Strategy for class weighting.
            random_state: Random seed for reproducibility.
        """
        self._model: Pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1,
            )),
        ])
        self._is_fitted: bool = False
        self._feature_names: list[str] = []

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
        """Return the names of features."""
        return self._feature_names.copy()

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int64],
        feature_names: list[str] | None = None,
    ) -> LevelClassifier:
        """Train the classifier.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).
            feature_names: Names of features.

        Returns:
            Self for method chaining.
        """
        self._model.fit(X, y)
        self._is_fitted = True
        self._feature_names = feature_names or [
            f"feature_{i}" for i in range(X.shape[1])
        ]
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        """Predict levels for the given features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted level indices.

        Raises:
            ValueError: If model is not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        return self._model.predict(X)

    def get_feature_importances(self) -> dict[str, float]:
        """Get feature importances from the trained model.

        Returns:
            Dictionary mapping feature names to importance scores.

        Raises:
            ValueError: If model is not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted.")

        classifier = self._model.named_steps["classifier"]
        importances = classifier.feature_importances_
        return dict(zip(self._feature_names, importances))
