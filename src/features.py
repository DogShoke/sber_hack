from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import ModelConfig


FEATURE_NAMES = [
    "answer_length_chars",
    "answer_length_tokens",
    "mean_logprob",
    "min_logprob",
    "std_logprob",
    "mean_entropy",
    "max_entropy",
]


@dataclass(slots=True)
class FeatureExtractionResult:
    feature_vector: np.ndarray
    feature_dict: dict[str, float]
    model_time_sec: float


def extract_uncertainty_features(
    token_logprobs: torch.Tensor,
    token_entropies: torch.Tensor,
) -> np.ndarray:
    if token_logprobs.numel() == 0:
        return np.zeros(5, dtype=np.float32)

    return np.asarray(
        [
            token_logprobs.mean().item(),
            token_logprobs.min().item(),
            token_logprobs.std(unbiased=False).item() if token_logprobs.numel() > 1 else 0.0,
            token_entropies.mean().item(),
            token_entropies.max().item(),
        ],
        dtype=np.float32,
    )


def extract_basic_text_features(answer_text: str, num_tokens: int) -> np.ndarray:
    normalized_answer = answer_text or ""
    return np.asarray(
        [
            float(len(normalized_answer)),
            float(num_tokens),
        ],
        dtype=np.float32,
    )


class GigaChatFeatureExtractor:
    def __init__(
        self,
        model_name_or_path: str | Path,
        config: ModelConfig | None = None,
    ) -> None:
        self.config = config or ModelConfig(model_name_or_path=str(model_name_or_path))
        self.model_name_or_path = str(model_name_or_path)
        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
        }
        # Newer Transformers versions may require this flag for Mistral-like regex handling.
        if "mistral" in self.model_name_or_path.lower() or "gigachat3.1" in self.model_name_or_path.lower():
            tokenizer_kwargs["fix_mistral_regex"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            **tokenizer_kwargs,
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.device_map:
            model_kwargs["device_map"] = self.config.device_map

        torch_dtype = _resolve_torch_dtype(self.config.torch_dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **model_kwargs,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def prepare_inputs(self, prompt: str, answer: str) -> tuple[torch.Tensor, int]:
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_encoding = self.tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        prompt_token_ids = _extract_input_ids(prompt_encoding)
        answer_start_idx = len(prompt_token_ids)

        full_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        full_encoding = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
        )
        full_token_ids = _extract_input_ids(full_encoding)

        token_ids = torch.tensor(
            [full_token_ids],
            dtype=torch.long,
            device=self.device,
        )
        return token_ids, answer_start_idx

    @torch.no_grad()
    def extract_features(self, prompt: str, answer: str) -> FeatureExtractionResult:
        token_ids, answer_start_idx = self.prepare_inputs(prompt=prompt, answer=answer)

        started_at = time.perf_counter()
        outputs = self.model(token_ids)
        model_time_sec = time.perf_counter() - started_at

        feature_vector = build_feature_vector(
            logits=outputs.logits,
            input_ids=token_ids,
            answer_start=answer_start_idx,
            answer_text=answer,
        )
        feature_dict = {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, feature_vector.tolist(), strict=True)
        }
        return FeatureExtractionResult(
            feature_vector=feature_vector,
            feature_dict=feature_dict,
            model_time_sec=model_time_sec,
        )


def build_feature_vector(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    answer_start: int,
    answer_text: str,
) -> np.ndarray:
    seq_len = input_ids.shape[1]
    answer_ids = input_ids[0, answer_start:seq_len]
    num_answer_tokens = int(answer_ids.numel())

    basic_text_features = extract_basic_text_features(
        answer_text=answer_text,
        num_tokens=num_answer_tokens,
    )

    if num_answer_tokens == 0:
        uncertainty_features = np.zeros(5, dtype=np.float32)
        return np.concatenate([basic_text_features, uncertainty_features]).astype(np.float32)

    answer_logits = logits[0, answer_start - 1 : seq_len - 1, :].float()
    log_probs = torch.log_softmax(answer_logits, dim=-1)
    token_logprobs = log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(-1)

    probs = torch.softmax(answer_logits, dim=-1)
    token_entropies = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

    uncertainty_features = extract_uncertainty_features(
        token_logprobs=token_logprobs,
        token_entropies=token_entropies,
    )
    return np.concatenate([basic_text_features, uncertainty_features]).astype(np.float32)


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "auto":
        return None
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Resolved object is not torch.dtype: {dtype_name}")
    return dtype


def _extract_input_ids(template_output: Any) -> list[int]:
    if isinstance(template_output, dict):
        return template_output["input_ids"]
    if isinstance(template_output, list):
        return template_output
    raise TypeError(f"Unsupported chat template output type: {type(template_output)!r}")
