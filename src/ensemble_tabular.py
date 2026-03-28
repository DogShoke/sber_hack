from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TrainConfig
from src.evaluate import build_precision_recall_curve, evaluate_predictions
from src.model_train import BaselineBundle
from src.utils import ensure_dir, load_pickle, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of cached tabular bundles.")
    parser.add_argument(
        "--feature-dump-path",
        type=Path,
        default=TrainConfig().feature_dump_path,
        help="Path to cached features created by src.extract_features.",
    )
    parser.add_argument(
        "--bundle-paths",
        type=Path,
        nargs="+",
        required=True,
        help="One or more trained bundle paths to ensemble.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=None,
        help="Optional ensemble weights matching --bundle-paths.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=TrainConfig().report_path,
        help="Where to save ensemble metrics.",
    )
    parser.add_argument(
        "--curve-path",
        type=Path,
        default=TrainConfig().curve_path,
        help="Where to save ensemble precision-recall curve.",
    )
    return parser.parse_args()


def predict_bundle(bundle: BaselineBundle, X: np.ndarray) -> np.ndarray:
    if bundle.scaler is not None:
        X_for_model = bundle.scaler.transform(X)
    else:
        X_for_model = pd.DataFrame(X, columns=bundle.feature_names)
    return bundle.classifier.predict_proba(X_for_model)[:, 1]


def main() -> None:
    args = parse_args()
    data = np.load(args.feature_dump_path, allow_pickle=True)
    X_val = data["X_val"]
    y_val = data["y_val"]
    val_times_ms = data["val_model_time_ms"]

    bundles = [load_pickle(bundle_path) for bundle_path in args.bundle_paths]
    weights = args.weights or [1.0] * len(bundles)
    if len(weights) != len(bundles):
        raise ValueError("Number of --weights must match number of --bundle-paths.")

    weights_array = np.asarray(weights, dtype=np.float64)
    weights_array = weights_array / weights_array.sum()

    bundle_scores = [predict_bundle(bundle, X_val) for bundle in bundles]
    ensemble_scores = np.average(np.vstack(bundle_scores), axis=0, weights=weights_array)

    metrics = evaluate_predictions(
        y_true=y_val,
        y_score=ensemble_scores,
        inference_times_ms=val_times_ms,
    )
    metrics.update(
        {
            "bundle_paths": [str(path) for path in args.bundle_paths],
            "weights": weights_array.tolist(),
            "num_validation_samples": int(len(y_val)),
            "num_features": int(X_val.shape[1]),
        }
    )

    curve_df = build_precision_recall_curve(y_true=y_val, y_score=ensemble_scores)
    ensure_dir(args.curve_path.parent)
    curve_df.to_csv(args.curve_path, index=False)
    save_json(metrics, args.report_path)

    print("Ensemble metrics:")
    print(pd.Series(metrics).to_string())
    print(f"\nCurve saved to: {args.curve_path}")
    print(f"Metrics saved to: {args.report_path}")


if __name__ == "__main__":
    main()
