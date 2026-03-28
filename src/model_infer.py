from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.config import ModelConfig
from src.features import GigaChatFeatureExtractor
from src.model_train import BaselineBundle
from src.utils import Timer, load_pickle


@dataclass(slots=True)
class InferenceResult:
    hallucination_probability: float
    model_time_ms: float
    overhead_time_ms: float
    total_time_ms: float


class HallucinationInferencePipeline:
    def __init__(self, bundle: BaselineBundle, extractor: GigaChatFeatureExtractor) -> None:
        self.bundle = bundle
        self.extractor = extractor

    def predict_one(self, prompt: str, model_answer: str) -> InferenceResult:
        forward_result = self.extractor.extract_features(prompt=prompt, answer=model_answer)

        with Timer() as timer:
            feature_vector = forward_result.feature_vector.reshape(1, -1)
            features_for_model = (
                self.bundle.scaler.transform(feature_vector)
                if self.bundle.scaler is not None
                else feature_vector
            )
            probability = float(self.bundle.classifier.predict_proba(features_for_model)[0, 1])

        return InferenceResult(
            hallucination_probability=probability,
            model_time_ms=forward_result.model_time_sec * 1000.0,
            overhead_time_ms=timer.elapsed * 1000.0,
            total_time_ms=(forward_result.model_time_sec + timer.elapsed) * 1000.0,
        )


def load_pipeline(
    model_bundle_path: str | Path,
    model_name_or_path: str | None = None,
) -> HallucinationInferencePipeline:
    bundle: BaselineBundle = load_pickle(model_bundle_path)
    model_source = model_name_or_path or bundle.model_config.model_name_or_path
    extractor = GigaChatFeatureExtractor(
        model_name_or_path=model_source,
        config=ModelConfig(model_name_or_path=model_source),
    )
    return HallucinationInferencePipeline(bundle=bundle, extractor=extractor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-example hallucination inference.")
    parser.add_argument("--model-bundle-path", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model-answer", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = load_pipeline(
        model_bundle_path=args.model_bundle_path,
        model_name_or_path=args.model_name_or_path,
    )
    result = pipeline.predict_one(prompt=args.prompt, model_answer=args.model_answer)

    print(f"hallucination_probability={result.hallucination_probability:.6f}")
    print(f"model_time_ms={result.model_time_ms:.2f}")
    print(f"overhead_time_ms={result.overhead_time_ms:.2f}")
    print(f"total_time_ms={result.total_time_ms:.2f}")


if __name__ == "__main__":
    main()
