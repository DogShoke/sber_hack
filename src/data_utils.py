from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DataConfig


REQUIRED_COLUMNS = ("prompt", "model_answer", "is_hallucination")


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return dataframe


def prepare_dataset(
    dataframe: pd.DataFrame,
    prompt_column: str = "prompt",
    answer_column: str = "model_answer",
    label_column: str = "is_hallucination",
) -> pd.DataFrame:
    dataset = dataframe[[prompt_column, answer_column, label_column]].copy()
    dataset = dataset.rename(
        columns={
            prompt_column: "prompt",
            answer_column: "model_answer",
            label_column: "is_hallucination",
        }
    )
    dataset["prompt"] = dataset["prompt"].fillna("").astype(str)
    dataset["model_answer"] = dataset["model_answer"].fillna("").astype(str)
    dataset["is_hallucination"] = (
        dataset["is_hallucination"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
    )
    if dataset["is_hallucination"].isna().any():
        bad_rows = dataset.index[dataset["is_hallucination"].isna()].tolist()[:10]
        raise ValueError(f"Unable to parse labels in rows: {bad_rows}")
    dataset["is_hallucination"] = dataset["is_hallucination"].astype(int)
    return dataset.reset_index(drop=True)


def load_and_prepare_dataset(config: DataConfig) -> pd.DataFrame:
    raw = load_dataset(config.data_path)
    return prepare_dataset(
        dataframe=raw,
        prompt_column=config.prompt_column,
        answer_column=config.answer_column,
        label_column=config.label_column,
    )


def split_train_validation(
    dataframe: pd.DataFrame,
    validation_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_labels = dataframe["is_hallucination"] if stratify else None
    train_df, validation_df = train_test_split(
        dataframe,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True)


def make_train_validation_split(config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_and_prepare_dataset(config)
    return split_train_validation(
        dataframe=dataset,
        validation_size=config.validation_size,
        random_state=config.random_state,
        stratify=config.stratify,
    )
