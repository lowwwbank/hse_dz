import argparse
import os
import sys
from pathlib import Path

from src.handlers.base import DataContext
from src.pipeline import create_default_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Process HH.ru resume data into numpy arrays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python app.py data/hh.csv
    python app.py /absolute/path/to/hh.csv

Output:
    Creates x_data.npy and y_data.npy in the same directory as the input file.

Features extracted:
    - gender: 1 (male), 0 (female), -1 (unknown)
    - age: Age in years
    - city_clean_encoded: Label-encoded city
    - experience_months: Total work experience in months
    - education_level: 0-4 ordinal scale
    - has_car_flag: 1 (has car), 0 (no car)

Target:
    - salary_numeric: Expected salary in rubles
        """,
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV file (hh.csv)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input file)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def validate_input_file(input_path: str) -> Path:
    """Validate that the input file exists and is readable.

    Args:
        input_path: Path to the input file.

    Returns:
        Validated Path object.

    Raises:
        SystemExit: If the file doesn't exist or isn't readable.
    """
    path = Path(input_path).resolve()

    if not path.exists():
        print(f"Error: Input file not found: {path}", file=sys.stderr)
        sys.exit(1)

    if not path.is_file():
        print(f"Error: Not a file: {path}", file=sys.stderr)
        sys.exit(1)

    if path.suffix.lower() != ".csv":
        print(f"Warning: Input file doesn't have .csv extension: {path}")

    return path


def main() -> int:
    """Main entry point for the application.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    # Validate input file
    input_path = validate_input_file(args.input_file)

    # Determine output directory
    output_dir = args.output_dir if args.output_dir else str(input_path.parent)
    output_path = Path(output_dir).resolve()

    if not output_path.exists():
        print(f"Creating output directory: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

    print(f"Input file: {input_path}")
    print(f"Output directory: {output_path}")
    print()

    # Create and execute pipeline
    try:
        pipeline = create_default_pipeline()
        context = DataContext(
            input_path=str(input_path),
            output_dir=str(output_path),
        )

        result = pipeline.execute(context)

        # Print summary
        print()
        print("Summary:")
        print(f"  Original shape: {result.metadata.get('original_shape', 'N/A')}")
        print(f"  Final shape: {result.metadata.get('final_shape', 'N/A')}")
        print(f"  Features: {result.metadata.get('feature_names', [])}")
        print()
        print(f"Output files created:")
        print(f"  - {result.metadata.get('x_path', 'N/A')}")
        print(f"  - {result.metadata.get('y_path', 'N/A')}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: Invalid data - {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error: Unexpected error - {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
