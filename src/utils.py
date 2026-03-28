from __future__ import annotations

import json
import pickle
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_pickle(obj: Any, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("wb") as file:
        pickle.dump(obj, file)
    return output_path


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def save_json(data: Any, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(to_serializable(data), file, ensure_ascii=False, indent=2)
    return output_path


def to_serializable(data: Any) -> Any:
    if is_dataclass(data):
        return to_serializable(asdict(data))
    if isinstance(data, dict):
        return {key: to_serializable(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_serializable(item) for item in data]
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (np.integer, np.floating)):
        return data.item()
    return data


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.perf_counter() - self.start
