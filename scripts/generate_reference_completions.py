from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from microtensor.core.hashing import digest_file  # noqa: E402
from microtensor.tasks.generator import MANIFEST_NAME  # noqa: E402


def load_train(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path} holds no tasks")
    return rows


def transformers_backend(model_spec: str):  # type: ignore[no-untyped-def]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "the transformers backend needs `pip install transformers torch`"
        ) from exc

    repo, _, revision = model_spec.partition("@")
    if not revision:
        raise SystemExit("pin the reference model as <repo>@<revision-sha>")

    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, torch_dtype="auto", device_map="auto"
    )
    model.eval()

    def complete(prompt: str, max_new_tokens: int) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    return complete


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one reference completion per public train task. Offline tooling: "
            "runs once per corpus version, never in the validator path."
        )
    )
    parser.add_argument("train", type=Path, help="code.train.jsonl from mt corpus generate")
    parser.add_argument("--model", required=True, help="<hf-repo>@<revision-sha>")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="stop after N tasks (smoke runs)")
    args = parser.parse_args()

    out = args.out or args.train.with_name("code.reference.jsonl")
    tasks = load_train(args.train)
    if args.limit:
        tasks = tasks[: args.limit]

    complete = transformers_backend(args.model)

    with out.open("w", encoding="utf-8") as fh:
        for index, task in enumerate(tasks, start=1):
            completion = complete(str(task["prompt"]), int(task.get("max_output_tokens", 512)))
            fh.write(
                json.dumps(
                    {"ref": task["ref"], "model": args.model, "completion": completion},
                    sort_keys=True,
                )
                + "\n"
            )
            if index % 25 == 0:
                print(f"{index}/{len(tasks)}", file=sys.stderr)

    digest = digest_file(out)
    print(f"{out.name}  {digest}")

    manifest_path = args.train.parent / MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][out.name] = digest
        manifest["reference_model"] = args.model
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"manifest updated: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
