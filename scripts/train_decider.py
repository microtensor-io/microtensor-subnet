from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microtensor.harness import decision_prompt
from microtensor.scoring.metrics import expected_answers, gold_label
from microtensor.tasks.corpus import load_corpus


@dataclass(frozen=True, slots=True)
class Example:
    context: str
    question: decision_prompt.Question
    target: list[float]


def _normalise(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _teacher(path: Path | None) -> dict[str, dict[str, dict[str, float]]]:
    if path is None:
        return {}
    found: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                found[str(row["ref"])] = row["answers"]
    return found


def _target(
    question: decision_prompt.Question, gold: Any, soft: dict[str, float] | None
) -> list[float]:
    labels = [_normalise(label) for label in question.labels]
    if soft:
        shares = [
            float(soft.get(label, soft.get(original, 0.0)))
            for label, original in zip(labels, question.labels, strict=True)
        ]
        total = math.fsum(shares)
        if total > 0.0:
            return [share / total for share in shares]
    answer = gold_label(question.kind, gold)
    if answer not in labels:
        return []
    return [1.0 if label == answer else 0.0 for label in labels]


def examples(corpus: Path, track: str, teacher: Path | None, seed: int) -> list[Example]:
    soft = _teacher(teacher)
    rng = random.Random(seed)
    found: list[Example] = []
    for task in load_corpus(corpus, track).tasks:
        spec = task.inputs.get("decision")
        if spec is None:
            continue
        shuffled = decision_prompt.shuffle_options(spec, f"train:{seed}:{task.ref}:{rng.random()}")
        context, questions = decision_prompt.parse(shuffled)
        expected = expected_answers(task.gold)
        for question in questions:
            if question.name not in expected:
                continue
            target = _target(
                question, expected[question.name], soft.get(task.ref, {}).get(question.name)
            )
            if target:
                found.append(Example(context=context, question=question, target=target))
    return found


def answer_ids(tokenizer: Any) -> dict[str, int]:
    found: dict[str, int] = {}
    for text in decision_prompt.ANSWER_STRINGS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(
                f"the tokenizer splits {text!r}; this base cannot enter a decision track"
            )
        found[text] = ids[0]
    if len(set(found.values())) != len(found):
        raise SystemExit("the tokenizer maps two answers to one token")
    return found


def render(tokenizer: Any, example: Example) -> str:
    return str(
        tokenizer.apply_chat_template(
            decision_prompt.messages(example.context, example.question),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def batch_loss(
    model: Any, tokenizer: Any, batch: Sequence[Example], ids: dict[str, int], brier: float
) -> Any:
    import torch
    import torch.nn.functional as functional

    texts = [render(tokenizer, example) for example in batch]
    encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    logits = model(**encoded).logits
    last = encoded["attention_mask"].sum(dim=1) - 1
    rows = logits[torch.arange(logits.size(0), device=logits.device), last]

    total = torch.zeros((), device=logits.device)
    for index, example in enumerate(batch):
        chosen = torch.tensor(
            [ids[text] for text in example.question.answers], device=logits.device
        )
        log_probs = functional.log_softmax(rows[index, chosen].float(), dim=-1)
        target = torch.tensor(example.target, device=logits.device)
        kl = functional.kl_div(log_probs, target, reduction="sum")
        squared = ((log_probs.exp() - target) ** 2).sum()
        total = total + kl + brier * squared
    return total / len(batch)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a calibrated decider on the public train split."
    )
    parser.add_argument("--base", required=True, help="Hugging Face base model id")
    parser.add_argument(
        "--revision", required=True, help="pinned base revision, as the allowlist records it"
    )
    parser.add_argument(
        "--corpus", type=Path, required=True, help="train split JSONL with decision tasks"
    )
    parser.add_argument("--track", default="classify")
    parser.add_argument(
        "--teacher", type=Path, help="JSONL of {ref, answers: {question: {label: p}}}"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--brier", type=float, default=1.0, help="weight of the Brier term; 0 is pure KL"
    )
    parser.add_argument("--full", action="store_true", help="full fine tune instead of LoRA")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=args.revision)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = answer_ids(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        revision=args.revision,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    if not args.full:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=args.rank,
                lora_alpha=args.rank * 2,
                lora_dropout=0.05,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )

    optimiser = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        data = examples(args.corpus, args.track, args.teacher, args.seed + epoch)
        if not data:
            raise SystemExit("the train split holds no decision question with a usable answer")
        random.Random(args.seed + epoch).shuffle(data)
        running = 0.0
        for start in range(0, len(data), args.batch):
            loss = batch_loss(model, tokenizer, data[start : start + args.batch], ids, args.brier)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += float(loss.detach())
        steps = max(1, math.ceil(len(data) / args.batch))
        print(f"epoch {epoch + 1}/{args.epochs}  questions {len(data)}  loss {running / steps:.4f}")

    if not args.full:
        model = model.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    keep = "--token-embedding-type q8_0" if tied else "--output-tensor-type q8_0"
    print(f"saved {args.out}")
    print("export:")
    print(f"  python convert_hf_to_gguf.py {args.out} --outfile decider-f16.gguf --outtype f16")
    print(f"  llama-quantize {keep} decider-f16.gguf decider.gguf Q4_K_M")
    print(
        "then calibrate: python scripts/fold_temperature.py --model decider.gguf "
        "--corpus <train> --out decider-cal.gguf"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
