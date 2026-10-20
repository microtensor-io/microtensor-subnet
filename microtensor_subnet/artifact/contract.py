from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final[int] = 1
COMPONENTS: Final[tuple[str, ...]] = ("slm", "harness", "router", "escalation")
HARDWARE_CLASSES: Final[tuple[str, ...]] = ("mt-1", "mt-2", "mt-3", "mt-4", "mt-5")
DEFAULT_HARDWARE_CLASS: Final[str] = "mt-5"

ARTIFACT_REPO_PREFIX: Final[str] = "microtensor_"
SLM_REPO_PREFIX: Final[str] = "mt_slm_"

MANIFEST_FILE: Final[str] = "manifest.json"
CONFIG_FILE: Final[str] = "config.json"
MAIN_FILE: Final[str] = "main.py"
REQUIREMENTS_FILE: Final[str] = "requirements.txt"
ROUTER_DIR: Final[str] = "router"
HARNESS_DIR: Final[str] = "harness"
ESCALATION_DIR: Final[str] = "escalation"
ROUTER_FILES: Final[tuple[str, ...]] = ("router/config.json", "router/model.safetensors")
HARNESS_FILES: Final[tuple[str, ...]] = ("harness/engine.py", "harness/verifier.py")
ESCALATION_FILES: Final[tuple[str, ...]] = ("escalation/patcher.py",)
CODE_DIRS: Final[tuple[str, ...]] = (HARNESS_DIR, ESCALATION_DIR)
CODE_LIMIT_BYTES: Final[int] = 2 * 1024 * 1024

SLM_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "README.md",
    "LICENSE",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
SLM_GGUF_REQUIRED_FILES: Final[tuple[str, ...]] = ("README.md", "LICENSE")
SLM_TRAINING_FILES: Final[tuple[str, ...]] = (
    "training/recipe.json",
    "training/data/sources.json",
)
SLM_METRICS_GLOB: Final[str] = "training/runs/*/metrics.jsonl"
SLM_MIN_PARAMETERS: Final[int] = 100_000_000
SLM_MAX_PARAMETERS: Final[int] = 8_000_000_000
ROUTER_MAX_PARAMETERS: Final[int] = 1_500_000_000
ROUTER_CPU_PARAMETERS: Final[int] = 100_000_000

ENV_PORT: Final[str] = "MT_PORT"
ENV_SLM_URL: Final[str] = "MT_SLM_URL"
ENV_ROUTER_URL: Final[str] = "MT_ROUTER_URL"
ENV_TOOLS_URL: Final[str] = "MT_TOOLS_URL"
ENV_RELAY_URL: Final[str] = "MT_RELAY_URL"

HARNESS_RUN: Final[str] = "/harness/run"
ESCALATION_REPAIR: Final[str] = "/escalation/repair"
SYSTEM_RUN: Final[str] = "/system/run"
ROUTER_DECIDE: Final[str] = "/decide"
SLM_COMPLETIONS: Final[str] = "/v1/completions"
RELAY_COMPLETIONS: Final[str] = "/v1/chat/completions"
TOOL_SCHEMA: Final[str] = "/tools/schema_lookup"
TOOL_QUERY: Final[str] = "/tools/query_execution"
TICKET_HEADER: Final[str] = "x-mt-ticket"
ROUTER_ANSWER: Final[str] = "draft_correct"

ALLOWED_PACKAGES: Final[dict[str, str]] = {
    "fastapi": "0.115.12",
    "uvicorn": "0.34.2",
    "starlette": "0.46.2",
    "pydantic": "2.11.4",
    "pydantic-core": "2.33.2",
    "httpx": "0.28.1",
    "httpcore": "1.0.9",
    "h11": "0.16.0",
    "anyio": "4.9.0",
    "sniffio": "1.3.1",
    "idna": "3.10",
    "certifi": "2025.4.26",
    "typing-extensions": "4.13.2",
    "annotated-types": "0.7.0",
    "click": "8.1.8",
    "orjson": "3.10.18",
    "numpy": "2.2.5",
    "regex": "2024.11.6",
    "jsonschema": "4.23.0",
    "sqlglot": "26.16.4",
    "sqlparse": "0.5.3",
    "python-dateutil": "2.9.0.post0",
    "six": "1.17.0",
    "pyyaml": "6.0.2",
    "tenacity": "9.1.2",
    "rapidfuzz": "3.13.0",
}

CONFIG_LIMITS: Final[dict[str, dict[str, int]]] = {
    "slm": {"max_context": 32_768, "max_new_tokens": 4_096},
    "router": {"max_input_tokens": 8_192},
    "harness": {"max_steps": 32, "max_total_tokens": 200_000},
}

DEFAULT_CONFIG: Final[dict[str, dict[str, object]]] = {
    "slm": {"max_context": 8192, "max_new_tokens": 1024},
    "router": {"max_input_tokens": 4096},
    "harness": {"max_steps": 8, "max_total_tokens": 32000},
    "escalation": {"preferred_model": ""},
}
