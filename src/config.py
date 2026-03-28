from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = ROOT_DIR / "docs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FEATURES_DIR = OUTPUTS_DIR / "features"
REPORTS_DIR = OUTPUTS_DIR / "reports"


@dataclass(slots=True)
class DataConfig:
    data_path: Path = DATA_DIR / "knowledge_bench_public.csv"
    prompt_column: str = "prompt"
    answer_column: str = "model_answer"
    label_column: str = "is_hallucination"
    validation_size: float = 0.3
    random_state: int = 42
    stratify: bool = True


@dataclass(slots=True)
class ModelConfig:
    model_name_or_path: str = "ai-sage/GigaChat3-10B-A1.8B-bf16"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True


@dataclass(slots=True)
class TrainConfig:
    max_iter: int = 2000
    class_weight: str = "balanced"
    seed: int = 42
    model_output_path: Path = MODELS_DIR / "logreg_baseline.pkl"
    feature_dump_path: Path = FEATURES_DIR / "baseline_features.npz"
    report_path: Path = REPORTS_DIR / "train_metrics.json"
    curve_path: Path = REPORTS_DIR / "precision_recall_curve.csv"
    predictions_path: Path = REPORTS_DIR / "validation_predictions.csv"
