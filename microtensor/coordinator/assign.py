from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

from microtensor.core.constants import COORDINATOR_REPLICATION


@dataclass(frozen=True, slots=True)
class Worker:
    hotkey: str
    device_profile: str = ""
    classes: tuple[str, ...] = ()

    def serves(self, hardware_class: str) -> bool:
        """Whether this worker has a certified device for the class.

        A worker with no declared classes serves all of them. That is the
        pre-certification state, and it keeps a fresh network assignable before
        any band is published rather than assigning nothing at all.
        """
        return not self.classes or hardware_class in self.classes


@dataclass(frozen=True, slots=True)
class System:
    digest: str
    track: str
    hardware_class: str
    miner_hotkey: str = ""
    source: str = ""

    @property
    def competition(self) -> tuple[str, str]:
        return self.track, self.hardware_class


def _rank(seed: str, system_digest: str, worker_hotkey: str) -> bytes:
    """Where one worker sorts for one system.

    Keyed hashing rather than a seeded RNG: a shuffle driven by a global
    generator depends on how many draws preceded it, so inserting one system
    would move every later assignment. This depends on nothing but its own
    three inputs, which is what makes the map recomputable from the seed.
    """
    material = f"{seed}\n{system_digest}\n{worker_hotkey}".encode()
    return sha256(material).digest()


def assign(
    systems: Sequence[System],
    workers: Sequence[Worker],
    seed: str,
    replication: int = COORDINATOR_REPLICATION,
) -> dict[str, tuple[str, ...]]:
    """Which workers measure which systems, deterministically in the inputs.

    Every system goes to `replication` workers so a single reading is never
    canonical and disagreement is detectable at all. Returns system digest to
    worker hotkeys.

    Publish the result. Anyone holding the seed and the metagraph can recompute
    this and check the coordinator assigned honestly, which costs one hash and
    removes the whole category of favouritism accusations.
    """
    if replication < 1:
        raise ValueError("replication must be at least one worker per system")

    out: dict[str, tuple[str, ...]] = {}

    for system in systems:
        eligible = [w for w in workers if w.serves(system.hardware_class)]
        if not eligible:
            out[system.digest] = ()
            continue
        ordered = sorted(eligible, key=lambda w: (_rank(seed, system.digest, w.hotkey), w.hotkey))
        out[system.digest] = tuple(w.hotkey for w in ordered[:replication])

    return out


def by_worker(assignment: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    """Invert the map so a worker can be handed its own list."""
    inverted: dict[str, list[str]] = {}
    for digest, hotkeys in assignment.items():
        for hotkey in hotkeys:
            inverted.setdefault(hotkey, []).append(digest)
    return {hotkey: tuple(sorted(digests)) for hotkey, digests in inverted.items()}


def under_replicated(
    assignment: dict[str, tuple[str, ...]], replication: int = COORDINATOR_REPLICATION
) -> tuple[str, ...]:
    """Systems that could not reach the replication target.

    Reported rather than hidden: a system measured by one worker is a system
    whose score nobody cross-checked, and the settlement should say so.
    """
    return tuple(sorted(d for d, w in assignment.items() if len(w) < replication))
