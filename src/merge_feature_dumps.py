from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple cached feature dumps into one.")
    parser.add_argument(
        "--input-paths",
        type=Path,
        nargs="+",
        required=True,
        help="Input .npz feature dumps to merge.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Where to save the merged .npz file.",
    )
    parser.add_argument(
        "--validation-source",
        type=str,
        choices=("first", "last", "all"),
        default="first",
        help="Which validation splits to keep in the merged dump.",
    )
    return parser.parse_args()


def _load_dump(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _assert_same_feature_names(dumps: list[dict[str, np.ndarray]]) -> list[str]:
    reference = dumps[0]["feature_names"].tolist()
    for dump in dumps[1:]:
        current = dump["feature_names"].tolist()
        if current != reference:
            raise ValueError("Feature names differ across input dumps; cannot merge safely.")
    return reference


def main() -> None:
    args = parse_args()
    dumps = [_load_dump(path) for path in args.input_paths]
    feature_names = _assert_same_feature_names(dumps)

    X_train = np.vstack([dump["X_train"] for dump in dumps]).astype(np.float32)
    y_train = np.concatenate([dump["y_train"] for dump in dumps]).astype(np.int32)
    train_times_ms = np.concatenate([dump["train_model_time_ms"] for dump in dumps]).astype(np.float32)

    if args.validation_source == "first":
        selected = [dumps[0]]
    elif args.validation_source == "last":
        selected = [dumps[-1]]
    else:
        selected = dumps

    X_val = np.vstack([dump["X_val"] for dump in selected]).astype(np.float32)
    y_val = np.concatenate([dump["y_val"] for dump in selected]).astype(np.int32)
    val_times_ms = np.concatenate([dump["val_model_time_ms"] for dump in selected]).astype(np.float32)

    ensure_dir(args.output_path.parent)
    np.savez_compressed(
        args.output_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_model_time_ms=train_times_ms,
        val_model_time_ms=val_times_ms,
        feature_names=np.asarray(feature_names),
    )

    print(f"Merged feature dump saved to: {args.output_path}")
    print(f"num_input_dumps={len(dumps)}")
    print(f"num_train_samples={len(y_train)}")
    print(f"num_validation_samples={len(y_val)}")
    print(f"num_features={X_train.shape[1]}")


if __name__ == "__main__":
    main()
