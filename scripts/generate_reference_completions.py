from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from microtensor.core.constants import PUBLIC_SERVER_URL  # noqa: E402
from microtensor.core.hashing import digest_file  # noqa: E402

MANIFEST_NAME = "corpus.manifest.json"


def load_train(path: Path) -> list[dict[str, object]]:
    """The train split from a local file: a bundle, or one task per line.

    Accepts either shape because the same tasks live in both: an upload
    bundle before it is sent, and the public split after it is published.
    """
    raw = path.read_text(encoding="utf-8")
    rows: list[dict[str, object]] = []

    stripped = raw.lstrip()
    if stripped.startswith("{") and "\n" in raw and '"tasks"' in raw[:2048]:
        payload = json.loads(raw)
        rows = [t for t in payload.get("tasks", []) if t.get("partition", "train") == "train"]
    else:
        for line in raw.splitlines():
            if line.strip():
                task = json.loads(line)
                if task.get("partition", "train") == "train":
                    rows.append(task)

    if not rows:
        raise SystemExit(f"{path} holds no train tasks")
    return rows


def fetch_train(api: str, version: str) -> list[dict[str, object]]:
    """The published train split, straight from the read API.

    The generator that used to write code.train.jsonl is gone; the train
    partition is served by the corpus endpoint now, so a reference set is
    produced against exactly what miners were given rather than against a
    file somebody kept a copy of.
    """
    import urllib.request

    url = f"{api.rstrip('/')}/v1/corpora/{version}/public"
    try:
        with urllib.request.urlopen(url, timeout=60) as answer:  # noqa: S310
            payload = json.loads(answer.read().decode("utf-8"))
    except Exception as exc:
        raise SystemExit(f"{url} could not be read: {exc}") from exc

    tasks = [t for t in payload.get("tasks", []) if t.get("partition") == "train"]
    if not tasks:
        raise SystemExit(f"corpus {version} publishes no train split")

    print(f"{len(tasks)} train tasks from {url}", file=sys.stderr)
    return tasks


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
    parser.add_argument(
        "train",
        type=Path,
        nargs="?",
        help="a bundle or train jsonl on disk; omit and pass --corpus-version instead",
    )
    parser.add_argument(
        "--corpus-version",
        help="published corpus version to read the train split from",
    )
    parser.add_argument("--api", default=PUBLIC_SERVER_URL, help="read API base URL")
    parser.add_argument("--model", required=True, help="<hf-repo>@<revision-sha>")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="stop after N tasks (smoke runs)")
    args = parser.parse_args()

    if not args.train and not args.corpus_version:
        raise SystemExit("pass a train file or --corpus-version")

    if args.train:
        tasks = load_train(args.train)
        default_out = args.train.with_name("code.reference.jsonl")
    else:
        tasks = fetch_train(args.api, args.corpus_version)
        default_out = Path("code.reference.jsonl")

    out = args.out or default_out
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

    manifest_path = (args.train.parent if args.train else out.parent) / MANIFEST_NAME
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
