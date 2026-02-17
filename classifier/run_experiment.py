#!/usr/bin/env python3
"""
IT Developer Level Classification - Proof of Concept

This script performs end-to-end classification experiment:
1. Load and filter IT developers from HH.ru dataset
2. Extract features and determine levels (junior/middle/senior)
3. Visualize class balance
4. Train classifier and evaluate
5. Generate classification report and conclusions

Usage:
    python run_experiment.py path/to/hh.csv
    python run_experiment.py path/to/hh.csv --output-dir ./results
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

from src.data_processor import DataProcessor
from src.level_classifier import LevelClassifier

# Use non-interactive backend so matplotlib doesn't open windows
plt.switch_backend("Agg")

MIN_SAMPLES = 20
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IT Developer Level Classification PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", type=str, help="Path to HH.ru CSV file")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="output",
        help="Output directory for plots and reports (default: output)",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Visualization helpers
# ------------------------------------------------------------------

def plot_class_balance(
    y: np.ndarray, labels: list[str], output_path: Path,
) -> dict[str, int]:
    """Plot class balance distribution (bar + pie chart)."""
    unique, counts = np.unique(y, return_counts=True)
    class_counts = {labels[i]: int(counts[idx]) for idx, i in enumerate(unique)}

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = axes[0].bar(class_counts.keys(), class_counts.values(), color=colors)
    axes[0].set_xlabel("Уровень разработчика", fontsize=12)
    axes[0].set_ylabel("Количество резюме", fontsize=12)
    axes[0].set_title("Распределение резюме по уровням", fontsize=14)

    for bar, count in zip(bars, class_counts.values()):
        axes[0].annotate(
            f"{count}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    axes[1].pie(
        class_counts.values(), labels=class_counts.keys(),
        autopct="%1.1f%%", colors=colors[: len(class_counts)],
        explode=[0.05] * len(class_counts), shadow=True,
    )
    axes[1].set_title("Доля каждого уровня", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved class balance plot: {output_path}")

    return class_counts


def plot_feature_importance(classifier: LevelClassifier, output_path: Path) -> None:
    """Plot horizontal bar chart of feature importances."""
    importances = classifier.get_feature_importances()
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    features, values = zip(*sorted_features)

    translations = {
        "gender": "Пол", "age": "Возраст", "salary": "Зарплата",
        "city_encoded": "Город", "experience_months": "Опыт (мес.)",
        "is_full_time": "Полная занятость", "education_level": "Образование",
        "has_car": "Авто",
    }
    features_ru = [translations.get(f, f) for f in features]

    plt.figure(figsize=(10, 6))
    bar_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
    bars = plt.barh(features_ru, values, color=bar_colors)
    plt.xlabel("Важность признака", fontsize=12)
    plt.title("Важность признаков для классификации", fontsize=14)
    plt.gca().invert_yaxis()

    for bar, val in zip(bars, values):
        plt.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved feature importance plot: {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output_path: Path,
) -> None:
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Предсказанный уровень", fontsize=12)
    plt.ylabel("Истинный уровень", fontsize=12)
    plt.title("Матрица ошибок", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix: {output_path}")


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------

def generate_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    class_counts: dict[str, int],
    cv_scores: np.ndarray,
    feature_importances: dict[str, float],
    output_path: Path,
) -> str:
    """Generate concise classification report."""
    report = classification_report(
        y_true, y_pred, target_names=labels, labels=[0, 1, 2], zero_division=0,
    )

    accuracy = float(np.mean(y_true == y_pred))
    total_samples = sum(class_counts.values())

    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

    feat_lines = "\n".join(
        f"  {name}: {imp:.4f}" for name, imp in sorted_features
    )

    text = f"""\
Classification Report
=====================

{report}
Accuracy: {accuracy:.4f}
CV ({CV_FOLDS}-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})

Class distribution (n={total_samples})
  junior: {class_counts.get('junior', 0)}
  middle: {class_counts.get('middle', 0)}
  senior: {class_counts.get('senior', 0)}

Feature importances
{feat_lines}
"""

    output_path.write_text(text, encoding="utf-8")
    print(f"  Saved report: {output_path}")
    return text


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    """Main entry point."""
    args = parse_args()

    input_path = Path(args.input_file).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IT Developer Level Classification - PoC")
    print("=" * 60)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_dir}\n")

    # Step 1: Load and process data
    print("Step 1: Loading and processing data...")
    processor = DataProcessor()

    try:
        df = processor.load_data(str(input_path))
        print(f"  Loaded {len(df):,} total resumes")

        df_it = processor.filter_it_developers(df)
        print(f"  Filtered {len(df_it):,} IT developers")

        if len(df_it) < MIN_SAMPLES:
            print(f"Error: Not enough IT developers ({len(df_it)}). Need >= {MIN_SAMPLES}.", file=sys.stderr)
            return 1

        X, y, feature_names = processor.extract_features(df_it)
        print(f"  Extracted {X.shape[1]} features from {X.shape[0]} labelled samples\n")
    except Exception as e:
        print(f"Error processing data: {e}", file=sys.stderr)
        return 1

    # Step 2: Visualize class balance
    print("Step 2: Visualizing class balance...")
    labels = ["junior", "middle", "senior"]
    class_counts = plot_class_balance(y, labels, output_dir / "class_balance.png")
    print()

    # Step 3: Train / test split
    print("Step 3: Training classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set:  {X_test.shape[0]} samples")

    classifier = LevelClassifier()
    classifier.fit(X_train, y_train, feature_names)
    print("  Training complete!")

    print("  Running cross-validation...")
    cv_scores = cross_val_score(classifier.model, X, y, cv=CV_FOLDS, scoring="accuracy")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")

    # Step 4: Evaluate
    print("Step 4: Evaluating model...")
    y_pred = classifier.predict(X_test)
    accuracy = float(np.mean(y_test == y_pred))
    print(f"  Test Accuracy: {accuracy:.4f}\n")

    # Step 5: Plots
    print("Step 5: Generating plots...")
    plot_feature_importance(classifier, output_dir / "feature_importance.png")
    plot_confusion_matrix(y_test, y_pred, labels, output_dir / "confusion_matrix.png")
    print()

    # Step 6: Report
    print("Step 6: Generating report...")
    report = generate_report(
        y_test, y_pred, labels, class_counts, cv_scores,
        classifier.get_feature_importances(), output_dir / "classification_report.txt",
    )
    print()
    print(report)

    print("=" * 60)
    print("Experiment completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
