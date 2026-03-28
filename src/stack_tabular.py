from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import TrainConfig
from src.evaluate import build_precision_recall_curve, evaluate_predictions
from src.model_train import fit_classifier
from src.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stacked tabular detector from cached features.")
    parser.add_argument(
        "--feature-dump-path",
        type=Path,
        default=TrainConfig().feature_dump_path,
        help="Path to cached features created by src.extract_features.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for OOF stacking.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=TrainConfig().report_path,
        help="Where to save stacked validation metrics.",
    )
    parser.add_argument(
        "--curve-path",
        type=Path,
        default=TrainConfig().curve_path,
        help="Where to save stacked precision-recall curve.",
    )
    return parser.parse_args()


def predict_with_bundle(
    scaler: StandardScaler | None,
    classifier: object,
    X: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    if scaler is not None:
        X_for_model = scaler.transform(X)
    else:
        X_for_model = pd.DataFrame(X, columns=feature_names)
    return classifier.predict_proba(X_for_model)[:, 1]


def main() -> None:
    args = parse_args()
    train_config = TrainConfig(report_path=args.report_path, curve_path=args.curve_path)

    data = np.load(args.feature_dump_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    val_times_ms = data["val_model_time_ms"]
    feature_names = data["feature_names"].tolist()

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=train_config.seed)
    oof_scores = np.zeros((len(y_train), 2), dtype=np.float32)

    for train_idx, valid_idx in cv.split(X_train, y_train):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_valid = X_train[valid_idx]

        logreg_scaler, logreg_clf = fit_classifier(
            X_train=X_fold_train,
            y_train=y_fold_train,
            train_config=train_config,
            classifier_name="logreg",
            feature_names=feature_names,
        )
        lgbm_scaler, lgbm_clf = fit_classifier(
            X_train=X_fold_train,
            y_train=y_fold_train,
            train_config=train_config,
            classifier_name="lightgbm",
            feature_names=feature_names,
        )

        oof_scores[valid_idx, 0] = predict_with_bundle(logreg_scaler, logreg_clf, X_fold_valid, feature_names)
        oof_scores[valid_idx, 1] = predict_with_bundle(lgbm_scaler, lgbm_clf, X_fold_valid, feature_names)

    meta_classifier = LogisticRegression(
        max_iter=3000,
        class_weight=train_config.class_weight,
        random_state=train_config.seed,
    )
    meta_classifier.fit(oof_scores, y_train)

    full_logreg_scaler, full_logreg_clf = fit_classifier(
        X_train=X_train,
        y_train=y_train,
        train_config=train_config,
        classifier_name="logreg",
        feature_names=feature_names,
    )
    full_lgbm_scaler, full_lgbm_clf = fit_classifier(
        X_train=X_train,
        y_train=y_train,
        train_config=train_config,
        classifier_name="lightgbm",
        feature_names=feature_names,
    )

    val_level0 = np.column_stack(
        [
            predict_with_bundle(full_logreg_scaler, full_logreg_clf, X_val, feature_names),
            predict_with_bundle(full_lgbm_scaler, full_lgbm_clf, X_val, feature_names),
        ]
    )
    stacked_scores = meta_classifier.predict_proba(val_level0)[:, 1]

    metrics = evaluate_predictions(
        y_true=y_val,
        y_score=stacked_scores,
        inference_times_ms=val_times_ms,
    )
    metrics.update(
        {
            "base_models": ["logreg", "lightgbm"],
            "cv_folds": args.cv_folds,
            "meta_classifier": "logreg",
            "num_train_samples": int(len(y_train)),
            "num_validation_samples": int(len(y_val)),
            "num_features": int(X_train.shape[1]),
            "feature_names": feature_names,
            "meta_coefficients": meta_classifier.coef_.tolist(),
            "meta_intercept": meta_classifier.intercept_.tolist(),
        }
    )

    curve_df = build_precision_recall_curve(y_true=y_val, y_score=stacked_scores)
    ensure_dir(train_config.curve_path.parent)
    curve_df.to_csv(train_config.curve_path, index=False)
    save_json(metrics, train_config.report_path)

    print("Stacked metrics:")
    print(pd.Series(metrics).to_string())
    print(f"\nCurve saved to: {train_config.curve_path}")
    print(f"Metrics saved to: {train_config.report_path}")


if __name__ == "__main__":
    main()
