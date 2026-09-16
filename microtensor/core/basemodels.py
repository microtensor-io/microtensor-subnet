from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class BaseModel:
    repo: str
    parameters: int
    derivation: str


HF_SAFETENSORS: Final[str] = "huggingface safetensors index, total parameter count"

BASE_MODELS: Final[dict[str, BaseModel]] = {
    "Qwen/Qwen2.5-0.5B-Instruct": BaseModel(
        "Qwen/Qwen2.5-0.5B-Instruct", 494_032_768, HF_SAFETENSORS
    ),
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": BaseModel(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct", 494_032_768, HF_SAFETENSORS
    ),
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": BaseModel(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", 1_543_714_304, HF_SAFETENSORS
    ),
    "Qwen/Qwen3-0.6B": BaseModel("Qwen/Qwen3-0.6B", 751_632_384, HF_SAFETENSORS),
    "Qwen/Qwen3-1.7B": BaseModel("Qwen/Qwen3-1.7B", 2_031_739_904, HF_SAFETENSORS),
    "Salesforce/xLAM-2-1b-fc-r": BaseModel(
        "Salesforce/xLAM-2-1b-fc-r", 1_543_714_304, HF_SAFETENSORS
    ),
    "google/gemma-3-4b-it": BaseModel("google/gemma-3-4b-it", 4_300_079_472, HF_SAFETENSORS),
    "meta-llama/Llama-3.2-3B-Instruct": BaseModel(
        "meta-llama/Llama-3.2-3B-Instruct", 3_212_749_824, HF_SAFETENSORS
    ),
    "microsoft/Phi-4-mini-instruct": BaseModel(
        "microsoft/Phi-4-mini-instruct", 3_836_021_760, HF_SAFETENSORS
    ),
    "Qwen/Qwen3.5-4B": BaseModel("Qwen/Qwen3.5-4B", 4_659_865_088, HF_SAFETENSORS),
    "Qwen/Qwen3.5-9B": BaseModel("Qwen/Qwen3.5-9B", 9_653_104_368, HF_SAFETENSORS),
}

MIN_BITS_PER_WEIGHT: Final[float] = 1.5
MIN_BITS_DERIVATION: Final[str] = (
    "IQ1_S, the smallest quantisation llama.cpp ships, averages about 1.56 bits "
    "per weight, and every other scheme in use is wider. A floor set below that "
    "cannot reject an artifact which genuinely holds the base model it declares."
)
UNKNOWN_FLOOR: Final[int] = 0


def repo_of(base_model: str) -> str:
    return base_model.split("@", 1)[0].strip()


def parameters_of(base_model: str) -> int:
    entry = BASE_MODELS.get(repo_of(base_model))
    return entry.parameters if entry is not None else 0


def size_floor_bytes(base_model: str) -> int:
    parameters = parameters_of(base_model)
    if parameters <= 0:
        return UNKNOWN_FLOOR
    return int(parameters * MIN_BITS_PER_WEIGHT / 8)
