from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microtensor.core.tracks import DECIDE, get_track
from microtensor.harness import decision_prompt
from microtensor.scoring.metrics import expected_answers, gold_label, score_task
from microtensor.tasks.corpus import load_corpus


@dataclass(frozen=True, slots=True)
class Generation:
    prompt: str
    completion: str


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
                answers = row.get("answers") or (row.get("teacher") or {}).get("answers") or {}
                found[str(row["ref"])] = {
                    name: dict(body.get("probabilities", body))
                    for name, body in answers.items()
                    if isinstance(body, dict)
                }
    return found


def _completions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    found: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row.get("completion"), str):
                    found[str(row["ref"])] = row["completion"]
    return found


def generations(corpus: Path, track: str, teacher: Path | None) -> list[Generation]:
    completions = _completions(teacher)
    return [
        Generation(prompt=task.prompt, completion=completions[task.ref])
        for task in load_corpus(corpus, track).tasks
        if task.ref in completions and completions[task.ref].strip()
    ]


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


def prompt_text(tokenizer: Any, prompt: str, chat: bool) -> str:
    if not chat:
        return prompt
    return str(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def generation_loss(model: Any, tokenizer: Any, batch: Sequence[Generation], chat: bool) -> Any:
    import torch

    total = torch.zeros((), device=model.device)
    for example in batch:
        head = tokenizer(prompt_text(tokenizer, example.prompt, chat), add_special_tokens=False)
        tail = tokenizer(example.completion + (tokenizer.eos_token or ""), add_special_tokens=False)
        ids = torch.tensor([head["input_ids"] + tail["input_ids"]], device=model.device)
        labels = ids.clone()
        labels[0, : len(head["input_ids"])] = -100
        total = total + model(input_ids=ids, labels=labels).loss
    return total / len(batch)


def rlcr_reward(score: float, confidence: float, weight: float) -> float:
    correct = 1.0 if score >= 0.5 else 0.0
    return score - weight * (confidence - correct) ** 2


def rlcr_step(
    model: Any,
    tokenizer: Any,
    tasks: Sequence[Any],
    metric: str,
    chat: bool,
    samples: int,
    temperature: float,
    weight: float,
) -> tuple[Any, float]:
    import torch

    total = torch.zeros((), device=model.device)
    rewards_seen: list[float] = []
    for task in tasks:
        head = tokenizer(prompt_text(tokenizer, task.prompt, chat), add_special_tokens=False)
        prompt_ids = torch.tensor([head["input_ids"]], device=model.device)
        with torch.no_grad():
            drawn = model.generate(
                prompt_ids,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=task.max_output_tokens,
                num_return_sequences=samples,
                pad_token_id=tokenizer.eos_token_id,
            )
        completions = drawn[:, prompt_ids.shape[1] :]
        log_means = []
        rewards = []
        for row in completions:
            kept = row[row != tokenizer.pad_token_id] if tokenizer.pad_token_id is not None else row
            if kept.numel() == 0:
                continue
            ids = torch.cat([prompt_ids[0], kept]).unsqueeze(0)
            logits = model(input_ids=ids).logits[0, prompt_ids.shape[1] - 1 : -1].float()
            chosen = torch.log_softmax(logits, dim=-1).gather(1, kept.unsqueeze(1)).squeeze(1)
            mean = chosen.mean()
            text = tokenizer.decode(kept, skip_special_tokens=True)
            score = float(score_task(metric, text, task.gold))
            rewards.append(rlcr_reward(score, float(mean.detach().exp()), weight))
            log_means.append(mean)
        if len(rewards) < 2:
            continue
        values = torch.tensor(rewards, device=model.device)
        advantage = (values - values.mean()) / (values.std() + 1e-6)
        total = total - (advantage * torch.stack(log_means)).mean()
        rewards_seen.extend(rewards)
    count = max(1, len(tasks))
    return total / count, (sum(rewards_seen) / len(rewards_seen) if rewards_seen else 0.0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train on the public train split of any arena: a calibrated decider for decision "
            "arenas, a supervised fine tune on teacher completions for the rest."
        )
    )
    parser.add_argument("--base", required=True, help="Hugging Face base model id")
    parser.add_argument(
        "--revision", required=True, help="pinned base revision, as the allowlist records it"
    )
    parser.add_argument("--corpus", type=Path, required=True, help="the arena's train split JSONL")
    parser.add_argument("--track", default="classify")
    parser.add_argument(
        "--teacher",
        type=Path,
        help="teacher JSONL from generate_reference_completions.py; required outside decisions",
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
    parser.add_argument(
        "--rlcr-steps", type=int, default=0, help="RL steps rewarding right and honest answers"
    )
    parser.add_argument("--rlcr-samples", type=int, default=4, help="answers drawn per task")
    parser.add_argument("--rlcr-temperature", type=float, default=0.8)
    parser.add_argument(
        "--rlcr-weight", type=float, default=1.0, help="weight of the calibration penalty"
    )
    parser.add_argument("--rlcr-batch", type=int, default=4)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    track = get_track(args.track)
    deciding = track.answer_mode == DECIDE
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=args.revision)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = answer_ids(tokenizer) if deciding else {}

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
        data: list[Any] = (
            list(examples(args.corpus, args.track, args.teacher, args.seed + epoch))
            if deciding
            else list(generations(args.corpus, args.track, args.teacher))
        )
        if not data:
            raise SystemExit("the train split holds nothing usable for this arena")
        random.Random(args.seed + epoch).shuffle(data)
        running = 0.0
        for start in range(0, len(data), args.batch):
            chunk = data[start : start + args.batch]
            loss = (
                batch_loss(model, tokenizer, chunk, ids, args.brier)
                if deciding
                else generation_loss(model, tokenizer, chunk, track.chat)
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += float(loss.detach())
        steps = max(1, math.ceil(len(data) / args.batch))
        print(f"epoch {epoch + 1}/{args.epochs}  examples {len(data)}  loss {running / steps:.4f}")

    if args.rlcr_steps and not deciding:
        pool = list(load_corpus(args.corpus, args.track).tasks)
        rng = random.Random(args.seed)
        for step in range(1, args.rlcr_steps + 1):
            batch = rng.sample(pool, min(args.rlcr_batch, len(pool)))
            loss, reward = rlcr_step(
                model,
                tokenizer,
                batch,
                track.metric,
                track.chat,
                args.rlcr_samples,
                args.rlcr_temperature,
                args.rlcr_weight,
            )
            if loss.requires_grad:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            print(f"rlcr {step}/{args.rlcr_steps}  mean reward {reward:.4f}")
    elif args.rlcr_steps:
        print("decision arenas already train on the Brier rule, which rewards honest confidence")

    if not args.full:
        model = model.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    keep = "--token-embedding-type q8_0" if tied else "--output-tensor-type q8_0"
    print(f"saved {args.out}")
    print("export:")
    print(f"  python convert_hf_to_gguf.py {args.out} --outfile model-f16.gguf --outtype f16")
    print(f"  llama-quantize {keep} model-f16.gguf model.gguf Q4_K_M")
    if deciding:
        print(
            "then calibrate: python scripts/fold_temperature.py --model model.gguf "
            "--corpus <train> --out model-cal.gguf"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
