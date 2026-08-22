from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

from microtensor.chain.config import ChainConfig
from microtensor.chain.metagraph import MetagraphSnapshot, snapshot_from
from microtensor.chain.weights import WeightVector, version_key
from microtensor.core.constants import (
    CHAIN_ATTEMPTS,
    CHAIN_BACKOFF_SECONDS,
    METAGRAPH_TTL_SECONDS,
)

log = logging.getLogger("microtensor.chain")

T = TypeVar("T")


class ChainError(RuntimeError):
    pass


# Named once, because three call sites hit it and a bare AttributeError from
# inside a retry wrapper tells a validator nothing about what to do. The
# dependency ceiling keeps a fresh install off 11; this is for a host that
# already had it.
UNADAPTED = (
    "this bittensor build exposes no {what}; microtensor supports bittensor "
    "9.x and 10.x, and 11 moved these onto namespaces the chain client does "
    "not use yet"
)


@runtime_checkable
class ChainClient(Protocol):
    @property
    def netuid(self) -> int: ...

    def block(self) -> int: ...

    def block_hash(self, block: int) -> str: ...

    def snapshot(self, *, refresh: bool = False) -> MetagraphSnapshot: ...

    def commitments(self, hotkeys: Sequence[str]) -> dict[str, str]: ...

    def publish(self, payload: str) -> bool: ...

    def set_weights(self, vector: WeightVector) -> tuple[bool, str]: ...

    def close(self) -> None: ...


def with_retry(
    label: str,
    call: Callable[[], T],
    *,
    attempts: int = CHAIN_ATTEMPTS,
    backoff: float = CHAIN_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            last = exc
            if attempt == attempts:
                break
            delay = backoff * 2 ** (attempt - 1)
            log.warning("%s failed (attempt %d/%d): %s", label, attempt, attempts, exc)
            sleep(delay)
    raise ChainError(f"{label} failed after {attempts} attempts: {last}") from last


class SubtensorClient:
    def __init__(self, config: ChainConfig, wallet: Any | None = None) -> None:
        self._config = config
        self._wallet = wallet
        self._subtensor: Any | None = None
        self._cached: MetagraphSnapshot | None = None
        self._cached_at = 0.0

    @property
    def netuid(self) -> int:
        return self._config.netuid

    @property
    def config(self) -> ChainConfig:
        return self._config

    @property
    def wallet(self) -> Any:
        if self._wallet is None:
            raise ChainError("this client is read only; construct it with a wallet to sign")
        return self._wallet

    @property
    def subtensor(self) -> Any:
        if self._subtensor is None:
            self._subtensor = self._connect()
        return self._subtensor

    def _connect(self) -> Any:
        try:
            import bittensor
        except ImportError as exc:
            raise ChainError(
                "bittensor is not installed; install the validator or miner extra"
            ) from exc

        factory = getattr(bittensor, "subtensor", None) or bittensor.Subtensor
        return with_retry(
            "connect",
            lambda: factory(network=self._config.resolved_endpoint),
        )

    def block(self) -> int:
        """The current block, from whichever surface this build exposes.

        bittensor 11 dropped get_current_block but kept `block` as a property,
        so both are tried for the same reason the metagraph read tries both.
        """
        reader = getattr(self.subtensor, "get_current_block", None)
        if callable(reader):
            return int(with_retry("get_current_block", reader))

        carried = getattr(self.subtensor, "block", None)
        if isinstance(carried, int):
            return int(carried)

        raise ChainError(UNADAPTED.format(what="block reader"))

    def block_hash(self, block: int) -> str:
        reader = getattr(self.subtensor, "get_block_hash", None)
        if not callable(reader):
            raise ChainError(UNADAPTED.format(what="block hash reader"))
        return str(with_retry("get_block_hash", lambda: reader(block)))

    def snapshot(self, *, refresh: bool = False) -> MetagraphSnapshot:
        age = time.monotonic() - self._cached_at
        fresh = self._cached is not None and age < METAGRAPH_TTL_SECONDS
        if self._cached is not None and fresh and not refresh:
            return self._cached

        raw = with_retry("metagraph", self._read_metagraph)
        snapshot = snapshot_from(raw, netuid=self._config.netuid)
        self._cached = snapshot
        self._cached_at = time.monotonic()
        return snapshot

    def _read_metagraph(self) -> Any:
        """Fetch the metagraph from whichever surface this build exposes.

        bittensor 11 moved it under a `subnets` namespace and dropped the method
        that used to sit on the client. Both are tried so a validator is not
        forced onto one release of a dependency to stay on the subnet.
        """
        namespace = getattr(self.subtensor, "subnets", None)
        reader = getattr(namespace, "metagraph", None)
        if callable(reader):
            found = reader(self._config.netuid)
            if found is None:
                raise ChainError(f"netuid {self._config.netuid} is not on this network")
            return found

        legacy = getattr(self.subtensor, "metagraph", None)
        if callable(legacy):
            return legacy(netuid=self._config.netuid)

        raise ChainError("this bittensor build exposes no metagraph reader")

    def commitments(self, hotkeys: Sequence[str]) -> dict[str, str]:
        carried = self._commitments_from_metagraph(hotkeys)
        if carried is not None:
            return carried

        snapshot = self.snapshot()
        uid_by_hotkey = snapshot.uid_by_hotkey
        reader = getattr(self.subtensor, "get_commitment", None)
        if reader is None:
            raise ChainError("this bittensor build exposes no commitment reader")

        found: dict[str, str] = {}
        for hotkey in hotkeys:
            uid = uid_by_hotkey.get(hotkey)
            if uid is None:
                continue
            try:
                raw = reader(netuid=self._config.netuid, uid=uid)
            except Exception as exc:
                log.warning("commitment read failed for uid %d: %s", uid, exc)
                continue
            if raw:
                found[hotkey] = str(raw)
        return found

    def _commitments_from_metagraph(self, hotkeys: Sequence[str]) -> dict[str, str] | None:
        """Read commitments off the metagraph when it already carries them.

        bittensor 11 returns them with the metagraph, which turns one read per
        miner into no extra reads at all. A build that does not carry them
        returns None so the per uid path still runs.
        """
        raw = with_retry("metagraph", self._read_metagraph)
        by_uid = getattr(raw, "commitments", None)
        unregistered = getattr(raw, "unregistered_commitments", None)
        if by_uid is None and unregistered is None:
            return None

        wanted = set(hotkeys)
        found: dict[str, str] = {}
        for record in getattr(raw, "neurons", ()) or ():
            hotkey = str(getattr(record, "hotkey", ""))
            if hotkey not in wanted:
                continue
            value = getattr(getattr(record, "commitment", None), "value", None)
            if value:
                found[hotkey] = str(value)

        for hotkey, commitment in (unregistered or {}).items():
            if str(hotkey) in wanted:
                value = getattr(commitment, "value", None)
                if value:
                    found.setdefault(str(hotkey), str(value))

        return found

    def publish(self, payload: str) -> bool:
        writer = getattr(self.subtensor, "commit", None) or getattr(
            self.subtensor, "set_commitment", None
        )
        if writer is None:
            raise ChainError("this bittensor build exposes no commitment writer")
        return bool(
            with_retry(
                "commit",
                lambda: writer(self.wallet, self._config.netuid, payload),
            )
        )

    def commit_reveal_enabled(self) -> bool:
        for name in ("commit_reveal_enabled", "get_subnet_reveal_period_epochs"):
            probe = getattr(self.subtensor, name, None)
            if probe is None:
                continue
            try:
                result = probe(netuid=self._config.netuid)
            except Exception as exc:
                log.debug("%s probe failed: %s", name, exc)
                continue
            if name == "commit_reveal_enabled":
                return bool(result)
            return bool(result) and int(result) > 0
        return False

    def set_weights(self, vector: WeightVector) -> tuple[bool, str]:
        if vector.is_empty:
            return False, "refusing to submit an empty weight vector"

        log.info(
            "submitting %d weights (commit-reveal %s)",
            len(vector),
            "on" if self.commit_reveal_enabled() else "off",
        )
        uids, values = vector.as_lists()

        submitted = self._set_weights_v11(uids, values)
        if submitted is not None:
            return submitted

        def call() -> Any:
            writer = getattr(self.subtensor, "set_weights", None)
            if not callable(writer):
                raise ChainError(UNADAPTED.format(what="set_weights"))
            return writer(
                wallet=self.wallet,
                netuid=self._config.netuid,
                uids=uids,
                weights=values,
                version_key=version_key(),
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )

        result = with_retry("set_weights", call)
        if isinstance(result, tuple) and len(result) == 2:
            return bool(result[0]), str(result[1])
        return bool(result), ""

    def _set_weights_v11(
        self, uids: Sequence[int], values: Sequence[int]
    ) -> tuple[bool, str] | None:
        """Submit through the module level helper bittensor 11 exposes.

        It takes proportions rather than the u16 vector, so the quantised
        values are divided back out. Quantising first is still what happens,
        because the vector that gets signed and the vector every worker
        recomputes have to be the same one.
        """
        try:
            import bittensor
        except ImportError:
            return None

        helper = getattr(bittensor, "set_weights", None)
        if not callable(helper) or hasattr(self.subtensor, "set_weights"):
            return None

        total = float(sum(values)) or 1.0
        paired = zip(uids, values, strict=True)
        proportions = {int(uid): float(value) / total for uid, value in paired}

        def call() -> Any:
            return helper(
                self._config.netuid,
                proportions,
                wallet=self.wallet,
                network=self._config.resolved_endpoint,
            )

        try:
            result = with_retry("set_weights", call)
        except ChainError as exc:
            return False, str(exc)

        if isinstance(result, tuple) and len(result) == 2:
            return bool(result[0]), str(result[1])
        return True if result is None else bool(result), ""

    def is_registered(self, hotkey: str) -> bool:
        return self.snapshot().is_registered(hotkey)

    def close(self) -> None:
        subtensor, self._subtensor = self._subtensor, None
        if subtensor is None:
            return
        closer = getattr(subtensor, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:
                log.debug("subtensor close failed: %s", exc)
