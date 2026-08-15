from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from microtensor.chain.client import ChainClient
from microtensor.chain.commitment import Commitment, build_commitment
from microtensor.chain.rounds import Round, round_for_block
from microtensor.core.constants import BLOCK_TIME_SECONDS, POLL_INTERVAL_SECONDS
from microtensor.miner.config import MinerConfig
from microtensor.miner.package import load_packaged
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
    return round_for_block(
        client.block(), length=config.round_blocks, genesis=config.genesis_block
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

    def stop(self, *_: object) -> None:
        self._running = False

    def step(self) -> Published | None:
        round_ = current_round(self.config, self.client)
        block = self.client.block()

        if any(p.round_index == round_.index for p in self.published):
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
