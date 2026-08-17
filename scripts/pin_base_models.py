from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CANDIDATES = (
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
)

API = "https://huggingface.co/api/models/{repo}"


def resolve(repo: str, *, revision: str, timeout: int) -> str:
    url = API.format(repo=repo) + f"/revision/{revision}"
    if not url.startswith("https://"):
        raise SystemExit(f"refusing a non-https model url: {url}")

    request = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": "microtensor-pin", "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise SystemExit(
                f"{repo} is gated; accept its licence on huggingface.co and export "
                "HF_TOKEN, or pin it by hand from the model's commit history"
            ) from exc
        raise SystemExit(f"{repo}: HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        raise SystemExit(f"{repo} could not be resolved: {exc}") from exc

    sha = str(payload.get("sha", ""))
    if len(sha) < 7:
        raise SystemExit(f"{repo} returned no usable commit sha")
    return sha


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve each candidate base model to its current commit sha and print the "
            "frozen ALLOWED_BASE_MODELS block. Freeze these the same day as the corpus."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="repeatable; overrides the candidate list",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    repos = args.model or list(CANDIDATES)
    pinned: list[str] = []
    for repo in repos:
        sha = resolve(repo, revision=args.revision, timeout=args.timeout)
        pinned.append(f"{repo}@{sha}")
        print(f"  {repo:<32} {sha}", file=sys.stderr)

    print(file=sys.stderr)
    print("ALLOWED_BASE_MODELS: Final[frozenset[str]] = frozenset({")
    for entry in sorted(pinned):
        print(f'    "{entry}",')
    print("})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
