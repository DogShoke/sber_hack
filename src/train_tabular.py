from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DataConfig, ModelConfig, TrainConfig
from src.evaluate import build_precision_recall_curve, evaluate_predictions
from src.model_train import BaselineBundle, fit_classifier
from src.utils import ensure_dir, load_pickle, save_json, save_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tabular classifier from cached features.")
    parser.add_argument(
        "--feature-dump-path",
        type=Path,
        default=TrainConfig().feature_dump_path,
        help="Path to cached features created by src.extract_features.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=("lightgbm", "logreg"),
        default="lightgbm",
        help="Tabular classifier trained on top of cached features.",
    )
    parser.add_argument(
        "--output-model-path",
        type=Path,
        default=TrainConfig().model_output_path,
        help="Where to save the trained model bundle.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=TrainConfig().report_path,
        help="Where to save validation metrics.",
    )
    parser.add_argument(
        "--curve-path",
        type=Path,
        default=TrainConfig().curve_path,
        help="Where to save precision-recall curve.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_config = TrainConfig(
        model_output_path=args.output_model_path,
        report_path=args.report_path,
        curve_path=args.curve_path,
    )

    data = np.load(args.feature_dump_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    train_times_ms = data["train_model_time_ms"]
    val_times_ms = data["val_model_time_ms"]
    feature_names = data["feature_names"].tolist()

    scaler, classifier = fit_classifier(
        X_train=X_train,
        y_train=y_train,
        train_config=train_config,
        classifier_name=args.classifier,
    )
    X_val_for_model = scaler.transform(X_val) if scaler is not None else X_val
    val_scores = classifier.predict_proba(X_val_for_model)[:, 1]

    metrics = evaluate_predictions(
        y_true=y_val,
        y_score=val_scores,
        inference_times_ms=val_times_ms,
    )
    metrics.update(
        {
            "classifier": args.classifier,
            "num_train_samples": int(len(y_train)),
            "num_validation_samples": int(len(y_val)),
            "num_features": int(X_train.shape[1]),
            "feature_names": feature_names,
            "train_model_time_ms_mean": float(train_times_ms.mean()),
            "validation_model_time_ms_mean": float(val_times_ms.mean()),
        }
    )

    bundle = BaselineBundle(
        feature_names=feature_names,
        classifier_name=args.classifier,
        scaler=scaler,
        classifier=classifier,
        data_config=DataConfig(),
        model_config=ModelConfig(),
        train_config=train_config,
    )

    curve_df = build_precision_recall_curve(y_true=y_val, y_score=val_scores)
    ensure_dir(train_config.curve_path.parent)
    curve_df.to_csv(train_config.curve_path, index=False)

    save_pickle(bundle, train_config.model_output_path)
    save_json(metrics, train_config.report_path)

    print("Validation metrics:")
    print(pd.Series(metrics).to_string())
    print(f"\nModel saved to: {train_config.model_output_path}")
    print(f"Curve saved to: {train_config.curve_path}")
    print(f"Metrics saved to: {train_config.report_path}")


if __name__ == "__main__":
    main()
