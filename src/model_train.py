from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import DataConfig, ModelConfig, TrainConfig
from src.data_utils import make_train_validation_split
from src.evaluate import build_precision_recall_curve, evaluate_predictions
from src.features import GigaChatFeatureExtractor
from src.utils import ensure_dir, save_json, save_pickle, set_seed


@dataclass(slots=True)
class BaselineBundle:
    feature_names: list[str]
    classifier_name: str
    scaler: StandardScaler | None
    classifier: Any
    data_config: DataConfig
    model_config: ModelConfig
    train_config: TrainConfig


def build_feature_matrix(
    dataframe: pd.DataFrame,
    extractor: GigaChatFeatureExtractor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_vectors: list[np.ndarray] = []
    labels: list[int] = []
    model_times_ms: list[float] = []

    iterator = tqdm(
        dataframe.itertuples(index=False),
        total=len(dataframe),
        desc="Extracting features",
    )
    for row in iterator:
        result = extractor.extract_features(prompt=row.prompt, answer=row.model_answer)
        feature_vectors.append(result.feature_vector)
        labels.append(int(row.is_hallucination))
        model_times_ms.append(result.model_time_sec * 1000.0)

    X = np.vstack(feature_vectors).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)
    times = np.asarray(model_times_ms, dtype=np.float32)
    return X, y, times


def fit_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_config: TrainConfig,
    classifier_name: str,
) -> tuple[StandardScaler | None, Any]:
    if classifier_name == "logreg":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        classifier = LogisticRegression(
            max_iter=train_config.max_iter,
            class_weight=train_config.class_weight,
            random_state=train_config.seed,
        )
        classifier.fit(X_train_scaled, y_train)
        return scaler, classifier

    if classifier_name == "lightgbm":
        classifier = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.5,
            objective="binary",
            random_state=train_config.seed,
            class_weight=train_config.class_weight,
            verbosity=-1,
        )
        classifier.fit(X_train, y_train)
        return None, classifier

    raise ValueError(f"Unsupported classifier: {classifier_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline hallucination detector.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DataConfig().data_path,
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=ModelConfig().model_name_or_path,
        help="Local path or HF id for GigaChat feature extraction.",
    )
    parser.add_argument(
        "--output-model-path",
        type=Path,
        default=TrainConfig().model_output_path,
        help="Where to save the trained model bundle.",
    )
    parser.add_argument(
        "--feature-dump-path",
        type=Path,
        default=TrainConfig().feature_dump_path,
        help="Where to save extracted features.",
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
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=TrainConfig().predictions_path,
        help="Where to save validation predictions.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples per split for quick smoke tests.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=("lightgbm", "logreg"),
        default="lightgbm",
        help="Tabular classifier trained on top of extracted features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_config = DataConfig(data_path=args.data_path)
    model_config = ModelConfig(model_name_or_path=args.model_name_or_path)
    train_config = TrainConfig(
        model_output_path=args.output_model_path,
        feature_dump_path=args.feature_dump_path,
        report_path=args.report_path,
        curve_path=args.curve_path,
        predictions_path=args.predictions_path,
    )

    set_seed(train_config.seed)
    train_df, validation_df = make_train_validation_split(data_config)
    if args.max_samples is not None:
        if args.max_samples < 2:
            raise ValueError("--max-samples must be at least 2")
        train_df = train_df.head(args.max_samples).reset_index(drop=True)
        validation_df = validation_df.head(args.max_samples).reset_index(drop=True)

    extractor = GigaChatFeatureExtractor(
        model_name_or_path=model_config.model_name_or_path,
        config=model_config,
    )

    X_train, y_train, train_times_ms = build_feature_matrix(train_df, extractor)
    X_val, y_val, val_times_ms = build_feature_matrix(validation_df, extractor)

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
            "num_train_samples": int(len(train_df)),
            "num_validation_samples": int(len(validation_df)),
            "num_features": int(X_train.shape[1]),
            "feature_names": extractor.feature_names,
            "train_model_time_ms_mean": float(train_times_ms.mean()),
            "validation_model_time_ms_mean": float(val_times_ms.mean()),
        }
    )

    bundle = BaselineBundle(
        feature_names=list(extractor.feature_names),
        classifier_name=args.classifier,
        scaler=scaler,
        classifier=classifier,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
    )

    ensure_dir(train_config.feature_dump_path.parent)
    np.savez_compressed(
        train_config.feature_dump_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_model_time_ms=train_times_ms,
        val_model_time_ms=val_times_ms,
        feature_names=np.asarray(extractor.feature_names),
    )

    curve_df = build_precision_recall_curve(y_true=y_val, y_score=val_scores)
    ensure_dir(train_config.curve_path.parent)
    curve_df.to_csv(train_config.curve_path, index=False)

    predictions_df = validation_df.copy()
    predictions_df["y_true"] = y_val
    predictions_df["y_score"] = val_scores
    predictions_df["inference_time_ms"] = val_times_ms
    ensure_dir(train_config.predictions_path.parent)
    predictions_df.to_csv(train_config.predictions_path, index=False)

    save_pickle(bundle, train_config.model_output_path)
    save_json(metrics, train_config.report_path)

    print("Validation metrics:")
    print(pd.Series(metrics).to_string())
    print(f"\nModel saved to: {train_config.model_output_path}")
    print(f"Features saved to: {train_config.feature_dump_path}")
    print(f"Curve saved to: {train_config.curve_path}")
    print(f"Predictions saved to: {train_config.predictions_path}")


if __name__ == "__main__":
    main()
