from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from microtensor_subnet.serving import markers


class TokenError(RuntimeError):
    pass


def _ids(found: Any) -> list[int]:
    if isinstance(found, Mapping) or hasattr(found, "keys"):
        found = found["input_ids"]
    if hasattr(found, "tolist"):
        found = found.tolist()
    rows = list(found or [])
    if rows and isinstance(rows[0], (list, tuple)):
        rows = list(rows[0])
    return [int(token) for token in rows]


class Tokeniser:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, model_dir: Path, *, template: str = "", gguf_file: str = "") -> Tokeniser:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise TokenError("transformers is required to count and return token ids") from exc
        options: dict[str, Any] = {"local_files_only": True, "trust_remote_code": False}
        if gguf_file and not (model_dir / "tokenizer_config.json").is_file():
            options["gguf_file"] = gguf_file
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), **options)
        except Exception as exc:
            raise TokenError(f"the tokeniser in {model_dir.name} did not load: {exc}") from exc
        if template:
            tokenizer.chat_template = (model_dir / template).read_text(encoding="utf-8")
        return cls(tokenizer)

    def chat_template_present(self) -> bool:
        return bool(getattr(self.tokenizer, "chat_template", None))

    def prompt_ids(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[int]:
        if not self.chat_template_present():
            raise TokenError("the model carries no chat template")
        found = self.tokenizer.apply_chat_template(
            [dict(m) for m in messages],
            tools=[dict(t) for t in tools] if tools else None,
            add_generation_prompt=True,
            tokenize=True,
        )
        return _ids(found)

    def completion_ids(
        self,
        family: str,
        *,
        reasoning: str,
        content: str,
        tool_calls: Sequence[Mapping[str, Any]],
        finish_reason: str = "",
    ) -> list[int]:
        text = markers.reassemble(family, reasoning, content, tool_calls)
        text += markers.end_of(family, finish_reason)
        if not text:
            return []
        return _ids(self.tokenizer(text, add_special_tokens=False))
