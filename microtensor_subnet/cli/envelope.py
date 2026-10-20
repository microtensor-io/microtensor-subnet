from __future__ import annotations

import argparse
import json
import secrets
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from microtensor_subnet.artifact.params import ParameterError, footprint_bytes, model_parameters
from microtensor_subnet.cli.common import add_common_arguments
from microtensor_subnet.core.constants import PUBLIC_SERVER_URL
from microtensor_subnet.scoring.stats import percentile
from microtensor_subnet.validator.config import LLAMA_IMAGE, VLLM_IMAGE
from microtensor_subnet.validator.context import render_prompt
from microtensor_subnet.validator.sandbox import Limits, cpu_container
from microtensor_subnet.validator.sandbox.docker import ContainerSpec, Docker, DockerError
from microtensor_subnet.validator.sandbox.slm import SlmClient, SlmError, vllm_command

PORT = 18000
READY_SECONDS = 900
TIMEOUT_SECONDS = 20
FILLER = "the agent opened the file, read the traceback and ran the tests again. "
CHARS_PER_TOKEN = 4


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("envelope", help="measure a model against an arena's envelope")
    inner = parser.add_subparsers(dest="action", required=True)
    check = inner.add_parser("check", help="run the model under the validators' limits and time it")
    check.add_argument("--arena", required=True)
    check.add_argument("--slm", type=Path, required=True, help="slm repository root")
    check.add_argument("--items", type=Path, help="items jsonl with a task per line")
    check.add_argument("--count", type=int, default=20, help="prompts to time")
    check.add_argument("--max-tokens", type=int, default=128, help="tokens to generate per prompt")
    check.add_argument("--gpu", default="0", help="GPU index for GPU groups")
    check.add_argument("--image", default="", help="override the vLLM or llama.cpp image")
    check.add_argument("--port", type=int, default=PORT)
    check.add_argument("--server", default=PUBLIC_SERVER_URL)
    add_common_arguments(check)
    check.set_defaults(handler=_check)


def _arena(server: str, arena_id: str) -> dict[str, Any]:
    url = f"{server.rstrip('/')}/v2/arenas/{arena_id}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as answer:  # noqa: S310
            found = json.loads(answer.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError) as exc:
        raise SystemExit(f"arena {arena_id} could not be read from {server}: {exc}") from exc
    return dict(found)


def _prompts(
    definition: Mapping[str, Any], items: Path | None, count: int, length: int
) -> list[str]:
    template = str(definition.get("slm_prompt") or "{user}")
    if items is not None:
        rows = [
            json.loads(line)
            for line in items.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [render_prompt(template, dict(row.get("task") or {})) for row in rows[:count]]
    filler = (FILLER * (length * CHARS_PER_TOKEN // len(FILLER) + 1))[: length * CHARS_PER_TOKEN]
    return [render_prompt(template, {"system": "", "user": filler})] * count


def _spec(
    args: argparse.Namespace, definition: Mapping[str, Any], limits: Limits, length: int
) -> ContainerSpec:
    name = f"mt-check-{secrets.token_hex(3)}"
    ports = ((f"127.0.0.1:{args.port}", 8000),)
    if limits.on_cpu:
        image = args.image or LLAMA_IMAGE
        return cpu_container(name, image, args.slm.resolve(), length, limits, "bridge", ports)
    return ContainerSpec(
        name=name,
        image=args.image or VLLM_IMAGE,
        network="bridge",
        command=vllm_command("/model", max_model_len=length),
        env={"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1"},
        volumes=((str(args.slm.resolve()), "/model", "ro"),),
        gpus=(args.gpu,),
        shm_size="8g",
        ports=ports,
    )


def _wait(client: SlmClient, docker: Docker, name: str) -> None:
    deadline = time.monotonic() + READY_SECONDS
    while time.monotonic() < deadline:
        if client.ready():
            return
        if not docker.running(name):
            raise SystemExit(f"the model server exited: {docker.logs(name, 20)[-600:]}")
        time.sleep(5.0)
    raise SystemExit("the model server did not become ready")


def _row(label: str, measured: str, limit: str, ok: bool) -> tuple[str, str, str, str]:
    return label, measured, limit, "pass" if ok else "FAIL"


def _check(args: argparse.Namespace) -> int:
    arena = _arena(args.server, args.arena)
    definition = dict(arena.get("definition") or {})
    budgets = dict(definition.get("budgets") or {})
    max_input = int(budgets.get("max_input_tokens") or 0)
    max_output = int(budgets.get("max_output_tokens") or 0)
    if not max_input or not max_output:
        raise SystemExit(f"arena {args.arena} names no envelope group")
    length = max_input + max_output
    limits = Limits.from_definition(definition, length)
    prompts = _prompts(definition, args.items, args.count, max_input)
    generation = definition.get("generation") or {}
    stop = [str(s) for s in generation.get("stop") or []]
    docker = Docker()
    spec = _spec(args, definition, limits, length)
    try:
        docker.start(spec)
    except DockerError as exc:
        raise SystemExit(str(exc)) from exc
    ttft: list[float] = []
    tpot: list[float] = []
    try:
        client = SlmClient(f"http://127.0.0.1:{args.port}")
        _wait(client, docker, spec.name)
        for prompt in prompts:
            try:
                made = client.generate(
                    prompt, max_tokens=min(args.max_tokens, max_output), stop=stop
                )
            except SlmError as exc:
                raise SystemExit(str(exc)) from exc
            ttft.append(made.ttft_ms)
            if made.completion_tokens > 1:
                tpot.append(made.tpot_ms)
        memory = int(docker.usage(spec.name).get("memory_peak_bytes") or 0) if limits.on_cpu else 0
    finally:
        docker.kill(spec.name)
    try:
        parameters = model_parameters(args.slm)
        memory = memory or footprint_bytes(args.slm, length)
    except ParameterError as exc:
        raise SystemExit(str(exc)) from exc
    rows = [
        _row(
            "parameters",
            f"{parameters:,}",
            f"{int(budgets.get('min_parameters') or 0):,} to {int(budgets['max_parameters']):,}",
            int(budgets.get("min_parameters") or 0) <= parameters <= int(budgets["max_parameters"]),
        ),
        _row(
            "memory",
            f"{memory / 1024**3:.2f} GB",
            f"{budgets['memory_gb']} GB",
            memory <= int(budgets["memory_bytes"]),
        ),
    ]
    first = percentile(ttft, 0.95)
    rows.append(
        _row(
            "first token p95",
            f"{first:,.0f} ms",
            f"{budgets['ttft_ms']:,} ms",
            first <= budgets["ttft_ms"],
        )
    )
    per = percentile(tpot, 0.95)
    if budgets.get("tpot_ms"):
        rows.append(
            _row(
                "per token p95",
                f"{per:.1f} ms",
                f"{budgets['tpot_ms']} ms",
                per <= budgets["tpot_ms"],
            )
        )
    if budgets.get("min_tokens_per_s"):
        rate = 1000.0 / per if per > 0 else 0.0
        floor = float(budgets["min_tokens_per_s"])
        rows.append(
            _row("generation", f"{rate:.1f} tokens/s", f"{floor:g} tokens/s", rate >= floor)
        )
    group = (definition.get("envelope") or {}).get("group", "")
    print(f"{args.arena} on {group}, {len(prompts)} prompts at up to {max_input:,} input tokens")
    for label, measured, limit, verdict in rows:
        print(f"  {label:<17} {measured:>16}   limit {limit:<26} {verdict}")
    return 0 if all(r[3] == "pass" for r in rows) else 1
