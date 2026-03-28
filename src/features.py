from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import ModelConfig


BASE_FEATURE_NAMES = [
    "answer_length_chars",
    "answer_length_tokens",
    "mean_logprob",
    "min_logprob",
    "std_logprob",
    "mean_entropy",
    "max_entropy",
    "num_sentences",
    "sentence_mean_logprob_mean",
    "sentence_mean_logprob_min",
    "sentence_mean_logprob_std",
    "sentence_max_entropy_mean",
    "sentence_max_entropy_max",
]
HIDDEN_FEATURE_STATS = ("mean", "std", "l2")
DEFAULT_LAYER_SPECS = ("mid", "late", "last")


@dataclass(slots=True)
class FeatureExtractionResult:
    feature_vector: np.ndarray
    feature_dict: dict[str, float]
    model_time_sec: float


def get_feature_names(layer_specs: tuple[str, ...] = DEFAULT_LAYER_SPECS) -> list[str]:
    feature_names = list(BASE_FEATURE_NAMES)
    for layer_name in layer_specs:
        for stat_name in HIDDEN_FEATURE_STATS:
            feature_names.append(f"hidden_{layer_name}_{stat_name}")
    for left_layer, right_layer in zip(layer_specs[:-1], layer_specs[1:], strict=True):
        feature_names.extend(
            [
                f"layer_cosine_distance_{left_layer}_vs_{right_layer}",
                f"layer_l2_distance_{left_layer}_vs_{right_layer}",
            ]
        )
    return feature_names


FEATURE_NAMES = get_feature_names()


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


def extract_sentence_uncertainty_features(
    token_logprobs: torch.Tensor,
    token_entropies: torch.Tensor,
    answer_text: str,
    num_answer_tokens: int,
) -> np.ndarray:
    sentence_spans = _approximate_sentence_token_spans(answer_text=answer_text, num_tokens=num_answer_tokens)
    if not sentence_spans:
        return np.zeros(6, dtype=np.float32)

    sentence_mean_logprobs: list[float] = []
    sentence_max_entropies: list[float] = []
    for start_idx, end_idx in sentence_spans:
        sentence_logprobs = token_logprobs[start_idx:end_idx]
        sentence_entropies = token_entropies[start_idx:end_idx]
        if sentence_logprobs.numel() == 0:
            continue
        sentence_mean_logprobs.append(float(sentence_logprobs.mean().item()))
        sentence_max_entropies.append(float(sentence_entropies.max().item()))

    if not sentence_mean_logprobs:
        return np.zeros(6, dtype=np.float32)

    mean_logprobs = np.asarray(sentence_mean_logprobs, dtype=np.float32)
    max_entropies = np.asarray(sentence_max_entropies, dtype=np.float32)
    return np.asarray(
        [
            float(len(sentence_mean_logprobs)),
            float(mean_logprobs.mean()),
            float(mean_logprobs.min()),
            float(mean_logprobs.std()),
            float(max_entropies.mean()),
            float(max_entropies.max()),
        ],
        dtype=np.float32,
    )


def extract_hidden_state_features(
    hidden_states: tuple[torch.Tensor, ...],
    answer_start: int,
    seq_len: int,
    layer_specs: tuple[str, ...] = DEFAULT_LAYER_SPECS,
) -> np.ndarray:
    if seq_len <= answer_start:
        num_hidden_features = len(layer_specs) * len(HIDDEN_FEATURE_STATS) + max(len(layer_specs) - 1, 0) * 2
        return np.zeros(num_hidden_features, dtype=np.float32)

    selected_layers = _select_hidden_layers(hidden_states=hidden_states, layer_specs=layer_specs)
    pooled_vectors = [_pool_answer_hidden_state(layer_tensor, answer_start, seq_len) for layer_tensor in selected_layers]

    feature_values: list[float] = []
    for pooled_vector in pooled_vectors:
        feature_values.extend(
            [
                float(pooled_vector.mean().item()),
                float(pooled_vector.std(unbiased=False).item()),
                float(torch.linalg.vector_norm(pooled_vector).item()),
            ]
        )

    for left_vector, right_vector in zip(pooled_vectors[:-1], pooled_vectors[1:], strict=True):
        cosine_similarity = torch.nn.functional.cosine_similarity(
            left_vector.unsqueeze(0),
            right_vector.unsqueeze(0),
        ).item()
        feature_values.extend(
            [
                float(1.0 - cosine_similarity),
                float(torch.linalg.vector_norm(left_vector - right_vector).item()),
            ]
        )

    return np.asarray(feature_values, dtype=np.float32)


class GigaChatFeatureExtractor:
    def __init__(
        self,
        model_name_or_path: str | Path,
        config: ModelConfig | None = None,
    ) -> None:
        self.config = config or ModelConfig(model_name_or_path=str(model_name_or_path))
        self.model_name_or_path = str(model_name_or_path)
        self.feature_names = get_feature_names()

        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
        }
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
        if self.config.low_cpu_mem_usage:
            model_kwargs["low_cpu_mem_usage"] = True
        if self.config.offload_folder is not None:
            model_kwargs["offload_folder"] = str(self.config.offload_folder)

        if self.config.load_in_4bit and self.config.load_in_8bit:
            raise ValueError("Only one quantization mode can be enabled at a time.")

        if self.config.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        torch_dtype = _resolve_torch_dtype(self.config.torch_dtype)
        if torch_dtype is not None and not (self.config.load_in_4bit or self.config.load_in_8bit):
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
        outputs = self.model(token_ids, output_hidden_states=True)
        model_time_sec = time.perf_counter() - started_at

        feature_vector = build_feature_vector(
            logits=outputs.logits,
            input_ids=token_ids,
            answer_start=answer_start_idx,
            answer_text=answer,
            hidden_states=outputs.hidden_states,
        )
        feature_dict = {
            name: float(value)
            for name, value in zip(self.feature_names, feature_vector.tolist(), strict=True)
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
    hidden_states: tuple[torch.Tensor, ...],
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
        sentence_features = np.zeros(6, dtype=np.float32)
        hidden_features = extract_hidden_state_features(
            hidden_states=hidden_states,
            answer_start=answer_start,
            seq_len=seq_len,
        )
        return np.concatenate([basic_text_features, uncertainty_features, sentence_features, hidden_features]).astype(
            np.float32
        )

    answer_logits = logits[0, answer_start - 1 : seq_len - 1, :].float()
    log_probs = torch.log_softmax(answer_logits, dim=-1)
    token_logprobs = log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(-1)

    probs = torch.softmax(answer_logits, dim=-1)
    token_entropies = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

    uncertainty_features = extract_uncertainty_features(
        token_logprobs=token_logprobs,
        token_entropies=token_entropies,
    )
    sentence_features = extract_sentence_uncertainty_features(
        token_logprobs=token_logprobs,
        token_entropies=token_entropies,
        answer_text=answer_text,
        num_answer_tokens=num_answer_tokens,
    )
    hidden_features = extract_hidden_state_features(
        hidden_states=hidden_states,
        answer_start=answer_start,
        seq_len=seq_len,
    )
    return np.concatenate([basic_text_features, uncertainty_features, sentence_features, hidden_features]).astype(
        np.float32
    )


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


def _approximate_sentence_token_spans(answer_text: str, num_tokens: int) -> list[tuple[int, int]]:
    normalized_text = (answer_text or "").strip()
    if not normalized_text or num_tokens <= 0:
        return []

    raw_sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", normalized_text) if segment.strip()]
    if not raw_sentences:
        return [(0, num_tokens)]

    total_chars = sum(max(len(sentence), 1) for sentence in raw_sentences)
    spans: list[tuple[int, int]] = []
    current_start = 0
    for index, sentence in enumerate(raw_sentences):
        remaining_sentences = len(raw_sentences) - index
        remaining_tokens = num_tokens - current_start
        if remaining_tokens <= 0:
            break
        if index == len(raw_sentences) - 1:
            token_count = remaining_tokens
        else:
            proportion = max(len(sentence), 1) / max(total_chars, 1)
            token_count = max(1, int(round(num_tokens * proportion)))
            token_count = min(token_count, remaining_tokens - max(remaining_sentences - 1, 0))
        current_end = min(num_tokens, current_start + token_count)
        spans.append((current_start, current_end))
        current_start = current_end
    return [(start, end) for start, end in spans if end > start]


def _select_hidden_layers(
    hidden_states: tuple[torch.Tensor, ...],
    layer_specs: tuple[str, ...],
) -> list[torch.Tensor]:
    if len(hidden_states) < 2:
        raise ValueError("Expected hidden states for multiple layers.")

    num_states = len(hidden_states)
    layer_indices: list[int] = []
    for layer_spec in layer_specs:
        if layer_spec == "mid":
            layer_indices.append(max(1, num_states // 2))
        elif layer_spec == "late":
            layer_indices.append(max(1, num_states - 3))
        elif layer_spec == "last":
            layer_indices.append(num_states - 1)
        else:
            raise ValueError(f"Unsupported layer spec: {layer_spec}")

    return [hidden_states[layer_index] for layer_index in layer_indices]


def _pool_answer_hidden_state(
    layer_hidden_state: torch.Tensor,
    answer_start: int,
    seq_len: int,
) -> torch.Tensor:
    answer_hidden_state = layer_hidden_state[0, answer_start:seq_len, :].float()
    if answer_hidden_state.shape[0] == 0:
        return torch.zeros(layer_hidden_state.shape[-1], dtype=torch.float32, device=layer_hidden_state.device)
    return answer_hidden_state.mean(dim=0)
