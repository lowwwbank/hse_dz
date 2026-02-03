#!/usr/bin/env python3
"""CLI application for salary prediction.

Loads a trained model and predicts salaries from input features.

Usage:
    python app.py path/to/x_data.npy

Output:
    List of predicted salaries in rubles (float).
"""

import argparse
import json
import sys
from pathlib import Path

from src.predictor import SalaryPredictor


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Predict salaries from HH.ru resume features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python app.py data/x_data.npy
    python app.py /path/to/features.npy --model custom_model.joblib
    python app.py data/x_data.npy --format json

Input format:
    .npy file with shape (n_samples, 6) containing features:
    - gender: 1 (male), 0 (female), -1 (unknown)
    - age: Age in years
    - city_clean_encoded: Label-encoded city
    - experience_months: Total work experience in months
    - education_level: 0-4 ordinal scale
    - has_car_flag: 1 (has car), 0 (no car)

Output:
    List of predicted salaries in rubles (float).
        """,
    )

    parser.add_argument(
        "input_file",
        type=str,
        nargs="?",
        help="Path to input .npy file with features",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Path to model file (default: resources/salary_model.joblib)",
    )

    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["plain", "json"],
        default="plain",
        help="Output format: 'plain' (one value per line) or 'json' (default: plain)",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show model information and exit",
    )

    return parser.parse_args()


def format_output(predictions: list[float], output_format: str) -> str:
    """Format predictions for output.

    Args:
        predictions: List of predicted salaries.
        output_format: Output format ('plain' or 'json').

    Returns:
        Formatted string.
    """
    if output_format == "json":
        return json.dumps(predictions, ensure_ascii=False)
    else:
        return "\n".join(str(p) for p in predictions)


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success).
    """
    args = parse_args()

    try:
        predictor = SalaryPredictor(model_path=args.model)

        if args.info:
            info = predictor.get_model_info()
            print("Model Information:")
            print(f"  Path: {info['model_path']}")
            print(f"  Fitted: {info['is_fitted']}")
            print(f"  Features: {info['feature_names']}")
            if info["metadata"]:
                print(
                    f"  Training samples: {info['metadata'].get('n_samples', 'N/A')}"
                )
                y_min = info["metadata"].get("y_min", 0)
                y_max = info["metadata"].get("y_max", 0)
                print(f"  Salary range: {y_min:.0f} - {y_max:.0f} руб.")
            return 0

        if not args.input_file:
            print("Error: input_file is required", file=sys.stderr)
            return 1

        input_path = Path(args.input_file).resolve()

        if not input_path.exists():
            print(f"Error: File not found: {input_path}", file=sys.stderr)
            return 1

        if input_path.suffix.lower() != ".npy":
            print(
                f"Error: Expected .npy file, got: {input_path.suffix}", file=sys.stderr
            )
            return 1

        predictions = predictor.predict_from_file(input_path)

        output = format_output(predictions, args.format)
        print(output)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: Invalid data - {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
