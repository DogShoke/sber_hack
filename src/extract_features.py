from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import DataConfig, ModelConfig, TrainConfig
from src.data_utils import make_train_validation_split
from src.model_train import build_feature_matrix
from src.features import GigaChatFeatureExtractor
from src.utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and cache hallucination features.")
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
        "--feature-dump-path",
        type=Path,
        default=TrainConfig().feature_dump_path,
        help="Where to save extracted features.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples per split for quick extraction tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_config = DataConfig(data_path=args.data_path)
    model_config = ModelConfig(model_name_or_path=args.model_name_or_path)
    train_config = TrainConfig(feature_dump_path=args.feature_dump_path)

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

    print(f"Features saved to: {train_config.feature_dump_path}")
    print(f"num_train_samples={len(train_df)}")
    print(f"num_validation_samples={len(validation_df)}")
    print(f"num_features={X_train.shape[1]}")


if __name__ == "__main__":
    main()
