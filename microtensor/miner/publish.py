from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from microtensor.chain.client import ChainClient
from microtensor.chain.commitment import Commitment, Reveal, build_commitment, commitment_hash, short_digest
from microtensor.chain.rounds import Round, round_for_block
from microtensor.core.constants import BLOCK_TIME_SECONDS, POLL_INTERVAL_SECONDS
from microtensor.miner.config import MinerConfig
from microtensor.miner.package import load_packaged, load_manifest_by_digest
from microtensor.registry.manifest import ArtifactManifest

log = logging.getLogger("microtensor.miner.publish")


class PublishError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Published:
    round_index: int
    commitment: Commitment
    payload: str

    @property
    def bytes_used(self) -> int:
        return len(self.payload.encode("utf-8"))


def current_round(config: MinerConfig, client: ChainClient) -> Round:
    """The round to submit to.

    When a coordinator is configured, its open round is authoritative: the
    operator sets the window, and computing it from block height instead would
    submit to the wrong round. Without a coordinator, or if it cannot be
    reached, the block schedule stands so a standalone miner still works.
    """
    served = _served_round(config)
    if served is not None:
        return served
    return round_for_block(
        client.block(), length=config.round_blocks, genesis=config.genesis_block
    )


def _served_round(config: MinerConfig) -> Round | None:
    if not config.coordinator_url:
        return None
    import json
    import urllib.request

    url = config.coordinator_url.rstrip("/") + "/v1/round/current"
    try:
        with urllib.request.urlopen(url, timeout=15) as answer:  # noqa: S310
            current = json.load(answer)
    except Exception as exc:
        log.warning("could not read the coordinator's round (%s); using the schedule", exc)
        return None

    index = current.get("round")
    close = current.get("close_block")
    start = current.get("start_block")
    end = current.get("end_block")
    if index is None or close is None or start is None or end is None:
        return None
    if current.get("phase") not in ("submissions", None):
        log.info("round %s is past submissions; nothing new to submit", index)
    return Round(
        index=int(index),
        start_block=int(start),
        length=int(end) - int(start),
        close_margin=int(end) - int(close),
    )


def commitment_for(
    config: MinerConfig, manifest: ArtifactManifest, round_index: int
) -> Commitment:
    return build_commitment(
        round_index,
        config.track,
        config.hardware_class,
        manifest.digest(),
        config.source,
        sealed=manifest.sealed is not None,
    )


def publish(
    config: MinerConfig,
    client: ChainClient,
    round_index: int,
    manifest: ArtifactManifest | None = None,
) -> Published:
    manifest = manifest or load_packaged(config)

    if manifest.round_index != round_index:
        raise PublishError(
            f"manifest was packaged for round {manifest.round_index}, not {round_index}; "
            "repackage before publishing"
        )
    if not manifest.signature:
        raise PublishError("manifest is unsigned; validators will reject it")

    commitment = commitment_for(config, manifest, round_index)
    payload = commitment.encode()

    if not client.publish(payload):
        raise PublishError("the chain rejected the commitment")

    log.info(
        "round %d: committed %d bytes pointing at %s",
        round_index,
        len(payload.encode("utf-8")),
        config.source,
    )
    return Published(round_index=round_index, commitment=commitment, payload=payload)


def reveal(
    config: MinerConfig, client: ChainClient, round_index: int, manifest: ArtifactManifest
) -> str:
    """Post the key for a sealed submission, replacing the slot's contents.

    The submission itself was already read by every party that needed it
    during the open window; from the close block onward the only thing the
    slot has to say is which artifact this is and how to open it.
    """
    from microtensor.miner.package import load_key

    if manifest.sealed is None:
        raise PublishError("this submission is not sealed; there is nothing to reveal")
    key = load_key(manifest.digest(), round_index)
    if key is None:
        raise PublishError(
            f"no key held for round {round_index}; was this artifact packaged here?"
        )
    payload = Reveal(
        round_index=round_index,
        manifest_digest=manifest.digest().split(":", 1)[-1][:32],
        key=key,
        commitment_hash=commitment_hash(commitment_for(config, manifest, round_index)),
    ).encode()
    if not client.publish(payload):
        raise PublishError("the chain rejected the reveal")
    log.info("round %d: key revealed", round_index)
    return payload


class PublishLoop:
    def __init__(
        self,
        config: MinerConfig,
        client: ChainClient,
        *,
        poll_seconds: int = POLL_INTERVAL_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.client = client
        self.poll_seconds = poll_seconds
        self._sleep = sleep
        self._running = False
        self.published: list[Published] = []
        self.revealed: set[int] = set()

    def stop(self, *_: object) -> None:
        self._running = False

    def step(self) -> Published | None:
        round_ = current_round(self.config, self.client)
        block = self.client.block()

        if any(p.round_index == round_.index for p in self.published):
            self._maybe_reveal(round_, block)
            self._sleep(self.poll_seconds)
            return None

        if not round_.accepts_submissions(block):
            remaining = round_.next.close_block - block
            log.info(
                "round %d already closed; next window opens in %d blocks",
                round_.index,
                round_.next.start_block - block,
            )
            self._sleep(min(self.poll_seconds, max(1, remaining * BLOCK_TIME_SECONDS)))
            return None

        manifest = load_packaged(self.config)
        if manifest.round_index != round_.index:
            log.warning(
                "manifest targets round %d but the chain is on %d; repackage to compete",
                manifest.round_index,
                round_.index,
            )
            self._sleep(self.poll_seconds)
            return None

        published = publish(self.config, self.client, round_.index, manifest)
        self.published.append(published)
        return published

    def _maybe_reveal(self, round_: Round, block: int) -> None:
        if round_.index in self.revealed or block < round_.close_block:
            return
        published = next((p for p in self.published if p.round_index == round_.index), None)
        if published is None:
            return
        manifest = load_manifest_by_digest(
            self.config, round_.index, published.commitment.manifest_digest
        )
        if manifest is None:
            try:
                manifest = load_packaged(self.config)
            except Exception:
                return
            if short_digest(manifest.digest()) != short_digest(
                published.commitment.manifest_digest
            ):
                log.warning(
                    "round %d: the packaged manifest is not the one committed on chain, "
                    "so no key can be revealed for it",
                    round_.index,
                )
                self.revealed.add(round_.index)
                return
        if manifest.sealed is None or manifest.round_index != round_.index:
            self.revealed.add(round_.index)
            return
        try:
            reveal(self.config, self.client, round_.index, manifest)
            self.revealed.add(round_.index)
        except PublishError as exc:
            log.warning("reveal not posted yet: %s", exc)

    def run(self, max_rounds: int | None = None) -> int:
        self._running = True
        log.info(
            "miner up on netuid %d in %s/%s",
            self.config.chain.netuid,
            self.config.track,
            self.config.hardware_class,
        )

        while self._running:
            if max_rounds is not None and len(self.published) >= max_rounds:
                break
            try:
                self.step()
            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as exc:
                log.exception("publish failed, retrying after backoff: %s", exc)
                self._sleep(self.poll_seconds)

        return len(self.published)
