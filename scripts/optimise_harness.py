from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from microtensor.core.hashing import digest_tree
from microtensor.core.protocol import ArtifactFormat, LoadManifest
from microtensor.core.system import EscalationRef
from microtensor.core.tracks import get_track
from microtensor.harness.engines.gguf import GgufEngine
from microtensor.harness.engines.router import Decision, ThresholdRouter
from microtensor.harness.package import load, scan
from microtensor.harness.sdk import (
    INPUT_SLOT,
    EscalationCall,
    Runtime,
    engine_small,
    openai_escalation,
)
from microtensor.scoring.metrics import score_task
from microtensor.tasks.corpus import Task, load_corpus

FENCE = re.compile(r"```(?:text)?\s*(.*?)```", re.DOTALL)
REFLECT = """You are improving the instructions a small language model follows for one task.

Current system prompt:
<system>
{system}
</system>

Current task prompt ({slot} is replaced by each input):
<task>
{task}
</task>

Here are inputs it got wrong, with its answer and the expected answer:
{failures}

Write an improved task prompt. Keep {slot} exactly once where the input goes. Be specific about
the output format and the mistakes above. Return only the new task prompt inside one ``` block."""


@dataclass(slots=True)
class Candidate:
    task: str
    scores: dict[str, float]

    @property
    def mean(self) -> float:
        return sum(self.scores.values()) / max(1, len(self.scores))


def _never(_: str) -> EscalationCall:
    raise RuntimeError("the harness search never escalates")


class Search:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.track = get_track(args.track)
        self.engine = GgufEngine()
        self.engine.load(
            args.model,
            LoadManifest(
                format=ArtifactFormat.GGUF,
                quantization="",
                entrypoint=args.model.name,
                max_input={"tokens": args.max_input_tokens},
            ),
        )
        self.teacher = openai_escalation(args.teacher_url, args.teacher_model, max_tokens=2048)
        self.spec = load(args.harness)
        prompts = dict(self.spec["prompts"])
        self.task_file = str(prompts.get("task", ""))
        self.system = (
            (args.harness / str(prompts["system"])).read_text(encoding="utf-8")
            if "system" in prompts
            else ""
        )
        self.work = Path(tempfile.mkdtemp(prefix="mt-harness-"))

    def package(self, task_prompt: str) -> Path:
        target = self.work / "candidate"
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(self.args.harness, target)
        if self.task_file:
            (target / self.task_file).write_text(task_prompt, encoding="utf-8")
        else:
            (target / "prompts").mkdir(exist_ok=True)
            (target / "prompts" / "task.txt").write_text(task_prompt, encoding="utf-8")
            spec = json.loads((target / "harness.json").read_text(encoding="utf-8"))
            spec["prompts"]["task"] = "prompts/task.txt"
            (target / "harness.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        return target

    def run(self, task_prompt: str, tasks: Sequence[Task]) -> dict[str, tuple[float, str]]:
        root = self.package(task_prompt)
        found: dict[str, tuple[float, str]] = {}
        for task in tasks:
            runtime = Runtime(
                root,
                ThresholdRouter((), Decision.RESOLVE),
                (),
                engine_small(
                    self.engine, chat=self.track.chat, max_output_tokens=task.max_output_tokens
                ),
                _never,
                escalation=EscalationRef(model="local/none", revision="0" * 40),
                system_digest="sha256:local",
                hotkey="local",
            )
            trace = runtime.run(0, task.ref, task.prompt, task.inputs)
            found[task.ref] = (
                score_task(self.track.metric, trace.final, task.gold),
                str(trace.final),
            )
        return found

    def reflect(self, task_prompt: str, failures: list[tuple[Task, str]]) -> str:
        shown = "\n\n".join(
            f"Input: {task.prompt[:1500]}\nAnswer: {answer[:500]}\n"
            f"Expected: {json.dumps(task.gold)[:500]}"
            for task, answer in failures
        )
        reply = self.teacher(
            REFLECT.format(system=self.system, task=task_prompt, failures=shown, slot=INPUT_SLOT)
        ).output
        blocks = FENCE.findall(reply)
        proposed = (blocks[0] if blocks else reply).strip()
        return proposed if proposed.count(INPUT_SLOT) == 1 else ""


def pareto(pool: Sequence[Candidate]) -> list[Candidate]:
    refs = set().union(*(c.scores for c in pool))
    front = []
    for candidate in pool:
        if any(
            candidate.scores.get(ref, 0.0) >= max(c.scores.get(ref, 0.0) for c in pool)
            for ref in refs
        ):
            front.append(candidate)
    return front or list(pool)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Search harness prompts: a frontier model reads the small model's failures and "
            "proposes better instructions, kept on a Pareto front across tasks."
        )
    )
    parser.add_argument("--model", type=Path, required=True, help="the small model GGUF")
    parser.add_argument("--harness", type=Path, required=True, help="the harness package")
    parser.add_argument("--corpus", type=Path, required=True, help="the arena's train split")
    parser.add_argument("--track", required=True)
    parser.add_argument("--teacher-url", required=True, help="OpenAI compatible endpoint")
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--out", type=Path, required=True, help="where to write the best package")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--minibatch", type=int, default=12)
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    search = Search(args)
    rng = random.Random(args.seed)
    tasks = list(load_corpus(args.corpus, args.track).tasks)
    rng.shuffle(tasks)
    validation, train = tasks[: args.validation], tasks[args.validation :] or tasks
    base_prompt = (
        (args.harness / search.task_file).read_text(encoding="utf-8")
        if search.task_file
        else INPUT_SLOT
    )
    base = Candidate(
        base_prompt, {r: s for r, (s, _) in search.run(base_prompt, validation).items()}
    )
    pool = [base]
    print(f"base harness: {base.mean:.3f} on {len(validation)} validation tasks")

    for iteration in range(1, args.iterations + 1):
        parent = rng.choice(pareto(pool))
        batch = rng.sample(train, min(args.minibatch, len(train)))
        ran = search.run(parent.task, batch)
        failures = [(t, ran[t.ref][1]) for t in batch if ran[t.ref][0] < 1.0]
        if not failures:
            continue
        child = search.reflect(parent.task, failures[:6])
        if not child:
            continue
        before = sum(s for s, _ in ran.values())
        after = sum(s for s, _ in search.run(child, batch).values())
        if after <= before:
            print(
                f"iteration {iteration}: proposal rejected on its minibatch ({after} <= {before})"
            )
            continue
        scored = Candidate(child, {r: s for r, (s, _) in search.run(child, validation).items()})
        pool.append(scored)
        print(f"iteration {iteration}: kept a candidate at {scored.mean:.3f}")

    best = max(pool, key=lambda c: c.mean)
    shutil.rmtree(args.out, ignore_errors=True)
    shutil.copytree(search.package(best.task), args.out)
    reason = scan(args.out)
    if reason:
        raise SystemExit(f"the best package fails the harness scan: {reason}")
    print(f"best harness: {best.mean:.3f} (base {base.mean:.3f}) from {len(pool)} candidates")
    print(f"wrote {args.out}  package_digest {digest_tree(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
