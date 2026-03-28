from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DataConfig, ModelConfig, TrainConfig
from src.evaluate import build_precision_recall_curve, evaluate_predictions
from src.model_train import BaselineBundle
from src.utils import ensure_dir, save_json, save_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune a tabular classifier on cached features.")
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
        help="Classifier family to tune.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for hyperparameter search.",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        choices=("random", "grid"),
        default="random",
        help="Search strategy. Random search is the practical default.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=32,
        help="Number of random-search iterations when --search-type=random.",
    )
    parser.add_argument(
        "--output-model-path",
        type=Path,
        default=TrainConfig().model_output_path,
        help="Where to save the tuned model bundle.",
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


def build_search_objects(
    classifier_name: str,
    seed: int,
    class_weight: str,
) -> tuple[object, dict[str, list[object]]]:
    if classifier_name == "lightgbm":
        estimator = LGBMClassifier(
            objective="binary",
            random_state=seed,
            class_weight=class_weight,
            verbosity=-1,
        )
        param_grid = {
            "n_estimators": [200, 400, 700],
            "learning_rate": [0.02, 0.03, 0.05],
            "num_leaves": [15, 31, 63],
            "min_child_samples": [10, 20, 40],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0.0, 0.1, 0.3],
            "reg_lambda": [0.0, 0.5, 1.0],
        }
        return estimator, param_grid

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight=class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )
    param_grid = {
        "classifier__C": [0.1, 0.3, 1.0, 3.0, 10.0],
        "classifier__solver": ["lbfgs", "liblinear"],
    }
    return pipeline, param_grid


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

    estimator, param_grid = build_search_objects(
        classifier_name=args.classifier,
        seed=train_config.seed,
        class_weight=train_config.class_weight,
    )

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=train_config.seed)
    X_train_search = pd.DataFrame(X_train, columns=feature_names) if args.classifier == "lightgbm" else X_train
    X_val_eval = pd.DataFrame(X_val, columns=feature_names) if args.classifier == "lightgbm" else X_val

    if args.search_type == "grid":
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
    else:
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=args.n_iter,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            random_state=train_config.seed,
        )
    search.fit(X_train_search, y_train)

    best_model = search.best_estimator_
    val_scores = best_model.predict_proba(X_val_eval)[:, 1]
    val_pr_auc = float(average_precision_score(y_val, val_scores))

    scaler = None
    classifier = best_model
    if args.classifier == "logreg":
        scaler = best_model.named_steps["scaler"]
        classifier = best_model.named_steps["classifier"]

    metrics = evaluate_predictions(
        y_true=y_val,
        y_score=val_scores,
        inference_times_ms=val_times_ms,
    )
    metrics.update(
        {
            "classifier": args.classifier,
            "cv_best_score_pr_auc": float(search.best_score_),
            "validation_pr_auc": val_pr_auc,
            "best_params": search.best_params_,
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

    print("Best params:")
    print(pd.Series(search.best_params_).to_string())
    print("\nValidation metrics:")
    print(pd.Series(metrics).to_string())
    print(f"\nModel saved to: {train_config.model_output_path}")
    print(f"Curve saved to: {train_config.curve_path}")
    print(f"Metrics saved to: {train_config.report_path}")


if __name__ == "__main__":
    main()
