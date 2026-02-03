#!/usr/bin/env python3
"""Training script for salary prediction model.

Trains a regression model on processed HH.ru resume data
and saves the model weights to the resources directory.

Usage:
    python train.py path/to/x_data.npy path/to/y_data.npy
    python train.py --x-path data/x_data.npy --y-path data/y_data.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

from src.model import SalaryRegressionModel


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train salary prediction model on HH.ru data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py data/x_data.npy data/y_data.npy
    python train.py --x-path data/x_data.npy --y-path data/y_data.npy -v

Output:
    Saves trained model to resources/salary_model.joblib
        """,
    )

    parser.add_argument(
        "x_path",
        type=str,
        nargs="?",
        help="Path to X features .npy file",
    )

    parser.add_argument(
        "y_path",
        type=str,
        nargs="?",
        help="Path to y target .npy file",
    )

    parser.add_argument(
        "--x-path",
        type=str,
        dest="x_path_opt",
        help="Path to X features .npy file (alternative)",
    )

    parser.add_argument(
        "--y-path",
        type=str,
        dest="y_path_opt",
        help="Path to y target .npy file (alternative)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for model (default: resources/salary_model.joblib)",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size for evaluation (default: 0.2)",
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )

    parser.add_argument(
        "--filter-outliers",
        action="store_true",
        help="Filter salary outliers before training",
    )

    parser.add_argument(
        "--salary-max",
        type=float,
        default=500_000,
        help="Maximum salary for outlier filtering (default: 500000)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def load_data(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load X and y data from .npy files.

    Args:
        x_path: Path to features file.
        y_path: Path to target file.

    Returns:
        Tuple of (X, y) numpy arrays.

    Raises:
        FileNotFoundError: If files don't exist.
    """
    if not x_path.exists():
        raise FileNotFoundError(f"X data file not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Y data file not found: {y_path}")

    X = np.load(x_path)
    y = np.load(y_path)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    return X, y


def filter_outliers(
    X: np.ndarray,
    y: np.ndarray,
    salary_min: float = 10_000,
    salary_max: float = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter salary outliers.

    Args:
        X: Feature matrix.
        y: Target vector.
        salary_min: Minimum valid salary.
        salary_max: Maximum valid salary.

    Returns:
        Filtered (X, y) tuple.
    """
    mask = (y >= salary_min) & (y <= salary_max)
    return X[mask], y[mask]


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    cv_folds: int = 5,
    verbose: bool = False,
) -> tuple[SalaryRegressionModel, dict]:
    """Train model and evaluate performance.

    Args:
        X: Feature matrix.
        y: Target vector (salaries in rubles).
        test_size: Proportion of data for test set.
        cv_folds: Number of cross-validation folds.
        verbose: Whether to print detailed output.

    Returns:
        Tuple of (trained model, evaluation metrics).
    """
    print(f"Dataset size: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"Salary range: {y.min():,.0f} - {y.max():,.0f} руб.")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    print()

    # Create and train model (with internal log transform)
    print("Training model (with log1p transform)...")
    model = SalaryRegressionModel(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        use_log_transform=True,
    )
    model.fit(X_train, y_train)
    print("Training complete!")
    print()

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print("Evaluation Results:")
    print(f"  Train R² score: {train_score:.4f}")
    print(f"  Test R² score:  {test_score:.4f}")
    print()

    # Cross-validation (on log-transformed internally)
    print(f"Cross-validation ({cv_folds} folds)...")

    # Custom CV scoring in original scale
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(X):
        X_cv_train, X_cv_val = X[train_idx], X[val_idx]
        y_cv_train, y_cv_val = y[train_idx], y[val_idx]

        cv_model = SalaryRegressionModel(use_log_transform=True)
        cv_model.fit(X_cv_train, y_cv_train)
        cv_scores.append(cv_model.score(X_cv_val, y_cv_val))

    cv_scores = np.array(cv_scores)
    print(f"  CV R² scores: {cv_scores}")
    print(f"  CV R² mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print()

    # Prediction error analysis
    y_pred = model.predict(X_test)
    errors = y_test - y_pred
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / y_test)) * 100

    print("Error Analysis (Test Set):")
    print(f"  MAE:  {mae:,.0f} руб.")
    print(f"  RMSE: {rmse:,.0f} руб.")
    print(f"  MAPE: {mape:.2f}%")
    print()

    # Feature importances
    if verbose:
        print("Feature Importances:")
        importances = model.get_feature_importances()
        for feature, importance in sorted(
            importances.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {feature}: {importance:.4f}")
        print()

    metrics = {
        "train_r2": train_score,
        "test_r2": test_score,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }

    return model, metrics


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success).
    """
    args = parse_args()

    x_path = args.x_path or args.x_path_opt
    y_path = args.y_path or args.y_path_opt

    if not x_path or not y_path:
        print("Error: Both x_path and y_path are required.", file=sys.stderr)
        print("Usage: python train.py path/to/x_data.npy path/to/y_data.npy")
        return 1

    x_path = Path(x_path).resolve()
    y_path = Path(y_path).resolve()

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = Path(__file__).parent / "resources" / "salary_model.joblib"

    print("=" * 60)
    print("Salary Prediction Model Training")
    print("=" * 60)
    print()
    print(f"X data: {x_path}")
    print(f"Y data: {y_path}")
    print(f"Output: {output_path}")
    print()

    try:
        X, y = load_data(x_path, y_path)
        print(f"Loaded {X.shape[0]:,} samples")

        # Filter outliers if requested
        if args.filter_outliers:
            print(f"Filtering outliers (salary <= {args.salary_max:,.0f})...")
            X, y = filter_outliers(X, y, salary_max=args.salary_max)
            print(f"After filtering: {X.shape[0]:,} samples")
        print()

        model, metrics = train_and_evaluate(
            X,
            y,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            verbose=args.verbose,
        )

        print(f"Saving model to {output_path}...")
        model.save(output_path)
        print("Model saved successfully!")
        print()

        print("=" * 60)
        print("Training completed successfully!")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
