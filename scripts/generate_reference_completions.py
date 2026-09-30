from __future__ import annotations

import argparse
import json
import os
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
    if not url.startswith(("http://", "https://")):
        raise SystemExit(f"--api needs an http or https URL, got {api!r}")

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


TOP_K = 8
DIGITS = 6


def _pinned(model_spec: str) -> tuple[str, str]:
    repo, _, revision = model_spec.partition("@")
    if not revision:
        raise SystemExit("pin the reference model as <repo>@<revision-sha>")
    return repo, revision


def _torch():  # type: ignore[no-untyped-def]
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit("teacher data needs `pip install transformers torch`") from exc
    return torch, transformers


def _causal(model_spec: str):  # type: ignore[no-untyped-def]
    torch, transformers = _torch()
    repo, revision = _pinned(model_spec)
    tokenizer = transformers.AutoTokenizer.from_pretrained(repo, revision=revision)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    return torch, tokenizer, model


def text_backend(model_spec: str):  # type: ignore[no-untyped-def]
    torch, tokenizer, model = _causal(model_spec)

    def teach(task: dict[str, object]) -> dict[str, object]:
        inputs = tokenizer(str(task["prompt"]), return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=int(task.get("max_output_tokens", 512)),
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        produced = output.sequences[0][inputs["input_ids"].shape[1] :]
        tokens, logprobs, top = [], [], []
        for token, scores in zip(produced.tolist(), output.scores, strict=False):
            log_p = torch.log_softmax(scores[0].float(), dim=-1)
            best = torch.topk(log_p, TOP_K)
            tokens.append(tokenizer.decode([token]))
            logprobs.append(round(float(log_p[token]), DIGITS))
            top.append(
                {
                    tokenizer.decode([int(i)]): round(float(v.exp()), DIGITS)
                    for v, i in zip(best.values, best.indices, strict=False)
                }
            )
        return {
            "completion": tokenizer.decode(produced, skip_special_tokens=True),
            "teacher": {"tokens": tokens, "logprobs": logprobs, "top": top},
        }

    return teach


def decide_backend(model_spec: str):  # type: ignore[no-untyped-def]
    from microtensor.core.tracks import DECISION_PROMPT_VERSION
    from microtensor.harness import decision_prompt as dp

    torch, tokenizer, model = _causal(model_spec)

    def answer_ids(question) -> list[int]:  # type: ignore[no-untyped-def]
        found = []
        for text in question.answers:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                raise SystemExit(f"the teacher splits the answer {text!r} into {len(ids)} tokens")
            found.append(ids[0])
        return found

    def teach(task: dict[str, object]) -> dict[str, object]:
        inputs = dict(task.get("inputs") or {})  # type: ignore[call-overload]
        context, questions = dp.parse(inputs.get("decision"))
        answers = {}
        for question in questions:
            rendered = tokenizer.apply_chat_template(
                dp.messages(context, question),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                logits = model(**encoded.to(model.device)).logits[0, -1].float()
            scores = [float(logits[i]) for i in answer_ids(question)]
            answers[question.name] = dp.build_answer(question, dp.to_grid(dp.softmax(scores)))
        return {"teacher": {"answers": answers, "prompt_format": DECISION_PROMPT_VERSION}}

    return teach


def _read_bytes(source: str) -> bytes:
    if source.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(source, timeout=120) as answer:  # noqa: S310
            return bytes(answer.read())
    return Path(source).read_bytes()


def speech_backend(model_spec: str):  # type: ignore[no-untyped-def]
    torch, transformers = _torch()
    from transformers.pipelines.audio_utils import ffmpeg_read

    repo, revision = _pinned(model_spec)
    processor = transformers.AutoProcessor.from_pretrained(repo, revision=revision)
    model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(repo, revision=revision)
    model.eval()
    rate = int(processor.feature_extractor.sampling_rate)

    def teach(task: dict[str, object]) -> dict[str, object]:
        inputs = dict(task.get("inputs") or {})  # type: ignore[call-overload]
        audio = ffmpeg_read(_read_bytes(str(inputs["audio"])), rate)
        features = processor(audio, sampling_rate=rate, return_tensors="pt").input_features
        with torch.no_grad():
            output = model.generate(features, output_scores=True, return_dict_in_generate=True)
        tail = output.sequences[0][-len(output.scores) :].tolist()
        words: list[dict[str, object]] = []
        for token, scores in zip(tail, output.scores, strict=False):
            if token in processor.tokenizer.all_special_ids:
                continue
            piece = processor.tokenizer.decode([token])
            probability = float(torch.softmax(scores[0].float(), dim=-1)[token])
            if piece.startswith(" ") or not words:
                words.append({"word": piece.strip(), "confidence": probability})
            else:
                words[-1]["word"] = str(words[-1]["word"]) + piece
                words[-1]["confidence"] = float(words[-1]["confidence"]) * probability  # type: ignore[arg-type]
        for word in words:
            word["confidence"] = round(float(word["confidence"]), DIGITS)  # type: ignore[arg-type]
        text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
        return {"completion": text, "teacher": {"transcript": text, "words": words}}

    return teach


def vision_backend(model_spec: str):  # type: ignore[no-untyped-def]
    _, transformers = _torch()
    repo, revision = _pinned(model_spec)
    detector = transformers.pipeline("object-detection", model=repo, revision=revision)
    label_ids = dict(detector.model.config.label2id)

    def teach(task: dict[str, object]) -> dict[str, object]:
        inputs = dict(task.get("inputs") or {})  # type: ignore[call-overload]
        found = detector(str(inputs["image"]), threshold=0.0)
        detections = [
            {
                "category_id": int(label_ids[d["label"]]),
                "bbox": [
                    round(float(d["box"]["xmin"]), 2),
                    round(float(d["box"]["ymin"]), 2),
                    round(float(d["box"]["xmax"] - d["box"]["xmin"]), 2),
                    round(float(d["box"]["ymax"] - d["box"]["ymin"]), 2),
                ],
                "score": round(float(d["score"]), DIGITS),
            }
            for d in found
            if d["label"] in label_ids
        ]
        return {"teacher": {"detections": detections}}

    return teach


def backend_for(track_id: str, model_spec: str):  # type: ignore[no-untyped-def]
    from microtensor.core.tracks import DECIDE, Modality, get_track

    track = get_track(track_id)
    if track.answer_mode == DECIDE:
        return decide_backend(model_spec)
    if track.modality is Modality.AUDIO:
        return speech_backend(model_spec)
    if track.modality is Modality.VISION and track.metric == "map_at_iou":
        return vision_backend(model_spec)
    if track.modality is Modality.TEXT:
        return text_backend(model_spec)
    raise SystemExit(f"no teacher backend for {track_id} ({track.modality.value}) yet")


def publish_reference(
    control: str, credential: str, version: str, model: str, rows: list[dict[str, object]]
) -> None:
    """Attach the completions to the corpus so miners can actually read them.

    Written locally first and posted second, so a run that dies halfway still
    leaves the expensive part on disk to retry from.
    """
    import urllib.request

    url = f"{control.rstrip('/')}/v1/operator/corpora/{version}/reference"
    if not url.startswith(("http://", "https://")):
        raise SystemExit(f"--publish needs an http or https URL, got {control!r}")

    body = json.dumps({"model": model, "completions": rows}).encode()
    request = urllib.request.Request(  # noqa: S310
        url,
        data=body,
        headers={"content-type": "application/json", "x-mt-credential": credential},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as answer:  # noqa: S310
            summary = json.loads(answer.read().decode("utf-8"))
    except Exception as exc:
        raise SystemExit(f"{url} refused the reference set: {exc}") from exc

    print(f"published: {summary.get('reference_count', 0)} completions on {version}")


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
    parser.add_argument(
        "--track",
        default="code",
        help="arena whose teacher data to produce; picks text, decision, speech or vision",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="stop after N tasks (smoke runs)")
    parser.add_argument(
        "--publish",
        metavar="CONTROL_URL",
        help="attach the result to the corpus, e.g. http://127.0.0.1:8081",
    )
    parser.add_argument(
        "--credential",
        default=os.environ.get("MTS_OPERATOR_SECRET", ""),
        help="operator credential; defaults to MTS_OPERATOR_SECRET",
    )
    args = parser.parse_args()

    if not args.train and not args.corpus_version:
        raise SystemExit("pass a train file or --corpus-version")

    if args.train:
        tasks = load_train(args.train)
        default_out = args.train.with_name(f"{args.track}.reference.jsonl")
    else:
        tasks = fetch_train(args.api, args.corpus_version)
        default_out = Path(f"{args.track}.reference.jsonl")

    out = args.out or default_out
    if args.limit:
        tasks = tasks[: args.limit]

    teach = backend_for(args.track, args.model)

    with out.open("w", encoding="utf-8") as fh:
        for index, task in enumerate(tasks, start=1):
            row = {"ref": task["ref"], "model": args.model, **teach(task)}
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            if index % 25 == 0:
                print(f"{index}/{len(tasks)}", file=sys.stderr)

    digest = digest_file(out)
    print(f"{out.name}  {digest}")

    if args.publish:
        if not args.corpus_version:
            raise SystemExit("--publish needs --corpus-version to attach to")
        if not args.credential:
            raise SystemExit("--publish needs an operator credential")
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
        publish_reference(args.publish, args.credential, args.corpus_version, args.model, rows)

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
