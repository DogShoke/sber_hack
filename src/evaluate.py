from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.utils import ensure_dir, save_json


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def build_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    threshold_values = np.append(thresholds, np.nan)
    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": threshold_values,
        }
    )


def summarize_inference_times(inference_times_ms: np.ndarray) -> dict[str, float]:
    return {
        "mean_inference_ms": float(np.mean(inference_times_ms)),
        "median_inference_ms": float(np.median(inference_times_ms)),
        "p95_inference_ms": float(np.percentile(inference_times_ms, 95)),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    inference_times_ms: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {"pr_auc": compute_pr_auc(y_true, y_score)}
    if inference_times_ms is not None and len(inference_times_ms) > 0:
        metrics.update(summarize_inference_times(inference_times_ms))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hallucination detector predictions.")
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=Path("outputs/reports/eval_metrics.json"))
    parser.add_argument("--curve-path", type=Path, default=Path("outputs/reports/eval_precision_recall_curve.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_df = pd.read_csv(args.predictions_path)

    required_columns = {"y_true", "y_score"}
    missing_columns = required_columns - set(predictions_df.columns)
    if missing_columns:
        raise ValueError(f"Predictions file is missing columns: {sorted(missing_columns)}")

    y_true = predictions_df["y_true"].to_numpy(dtype=int)
    y_score = predictions_df["y_score"].to_numpy(dtype=float)
    inference_times_ms = None
    if "inference_time_ms" in predictions_df.columns:
        inference_times_ms = predictions_df["inference_time_ms"].to_numpy(dtype=float)

    metrics = evaluate_predictions(
        y_true=y_true,
        y_score=y_score,
        inference_times_ms=inference_times_ms,
    )
    curve_df = build_precision_recall_curve(y_true=y_true, y_score=y_score)

    ensure_dir(args.curve_path.parent)
    curve_df.to_csv(args.curve_path, index=False)
    save_json(metrics, args.report_path)

    print(pd.Series(metrics).to_string())
    print(f"\nCurve saved to: {args.curve_path}")
    print(f"Metrics saved to: {args.report_path}")


if __name__ == "__main__":
    main()
