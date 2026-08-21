from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_sums(directory: Path, patterns: list[str]) -> Path:
    files = sorted(
        {p for pattern in patterns for p in directory.glob(pattern) if p.is_file()},
        key=lambda p: p.name,
    )
    if not files:
        raise SystemExit(f"no artifacts matching {patterns} under {directory}")

    sums = directory / "SHA256SUMS"
    sums.write_text("\n".join(f"{sha256(p)}  {p.name}" for p in files) + "\n", encoding="utf-8")
    for line in sums.read_text(encoding="utf-8").splitlines():
        print(line)
    return sums


# The substrate crypto type for ed25519. An integer, not a KeypairType
# member: bittensor_wallet exports no such name, so asking for one sent every
# signing attempt down the fallback and failed on hosts without
# substrateinterface — which is what stopped the first release.
ED25519 = 0


def _keypair_class() -> Any:
    try:
        from bittensor_wallet import Keypair as WalletKeypair

        return WalletKeypair
    except ImportError:
        pass

    try:
        from substrateinterface import Keypair as SubstrateKeypair

        return SubstrateKeypair
    except ImportError as exc:
        raise SystemExit(
            "no keypair implementation is installed; pip install bittensor-wallet"
        ) from exc


def sign(sums: Path, seed_hex: str) -> Path:
    keypair_class = _keypair_class()
    keypair = keypair_class.create_from_seed(
        seed_hex if seed_hex.startswith("0x") else f"0x{seed_hex}",
        crypto_type=ED25519,
    )
    signature = keypair.sign(sums.read_bytes())
    destination = sums.with_suffix(sums.suffix + ".sig")
    destination.write_bytes(signature)

    print(f"\npublic key  0x{keypair.public_key.hex()}")
    print(f"signature   {destination.name} ({len(signature)} bytes)")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Write and sign SHA256SUMS for a release.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pattern", action="append", default=None)
    parser.add_argument(
        "--seed-env",
        default="MT_RELEASE_SIGNING_SEED",
        help="environment variable holding the hex ed25519 seed",
    )
    parser.add_argument("--unsigned", action="store_true", help="write SHA256SUMS only")
    args = parser.parse_args()

    sums = write_sums(args.directory, args.pattern or ["*.whl", "*.tar.gz"])
    if args.unsigned:
        print("\nWARNING: unsigned release; validators requiring a signature will refuse it")
        return 0

    seed = os.environ.get(args.seed_env, "").strip()
    if not seed:
        print(f"error: {args.seed_env} is not set", file=sys.stderr)
        return 1

    sign(sums, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
