from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from microtensor.core.constants import MIN_VALIDATOR_STAKE


@dataclass(frozen=True, slots=True)
class Neuron:
    uid: int
    hotkey: str
    coldkey: str = ""
    stake: float = 0.0
    trust: float = 0.0
    validator_trust: float = 0.0
    incentive: float = 0.0
    emission: float = 0.0
    validator_permit: bool = False
    active: bool = True
    last_update: int = 0
    address: str = ""
    port: int = 0

    @property
    def is_validator(self) -> bool:
        return self.validator_permit and self.stake >= MIN_VALIDATOR_STAKE

    @property
    def is_serving(self) -> bool:
        return bool(self.address) and self.port > 0


@dataclass(frozen=True, slots=True)
class MetagraphSnapshot:
    netuid: int
    block: int
    neurons: tuple[Neuron, ...] = ()

    def __post_init__(self) -> None:
        seen: set[int] = set()
        for neuron in self.neurons:
            if neuron.uid in seen:
                raise ValueError(f"duplicate uid {neuron.uid} in metagraph snapshot")
            seen.add(neuron.uid)

    def __len__(self) -> int:
        return len(self.neurons)

    @property
    def hotkeys(self) -> tuple[str, ...]:
        return tuple(n.hotkey for n in self.neurons)

    @property
    def uid_by_hotkey(self) -> dict[str, int]:
        return {n.hotkey: n.uid for n in self.neurons}

    @property
    def total_stake(self) -> float:
        return sum(n.stake for n in self.neurons)

    def find(self, hotkey: str) -> Neuron | None:
        for neuron in self.neurons:
            if neuron.hotkey == hotkey:
                return neuron
        return None

    def at(self, uid: int) -> Neuron | None:
        for neuron in self.neurons:
            if neuron.uid == uid:
                return neuron
        return None

    def is_registered(self, hotkey: str) -> bool:
        return self.find(hotkey) is not None

    def uid_of(self, hotkey: str) -> int:
        neuron = self.find(hotkey)
        if neuron is None:
            raise KeyError(f"hotkey {hotkey!r} is not registered on netuid {self.netuid}")
        return neuron.uid

    def validators(self, min_stake: float = MIN_VALIDATOR_STAKE) -> tuple[Neuron, ...]:
        return tuple(
            n for n in self.neurons if n.validator_permit and n.stake >= min_stake
        )

    def miners(self, min_stake: float = MIN_VALIDATOR_STAKE) -> tuple[Neuron, ...]:
        validators = {n.uid for n in self.validators(min_stake)}
        return tuple(n for n in self.neurons if n.uid not in validators)

    def addresses(self) -> dict[str, str]:
        return {n.hotkey: n.address for n in self.neurons if n.address}

    def coldkeys(self) -> dict[str, str]:
        return {n.hotkey: n.coldkey for n in self.neurons if n.coldkey}

    def stake_share(self, hotkey: str) -> float:
        total = self.total_stake
        neuron = self.find(hotkey)
        if neuron is None or total <= 0.0:
            return 0.0
        return neuron.stake / total

    def has_permit(self, hotkey: str, min_stake: float = MIN_VALIDATOR_STAKE) -> bool:
        neuron = self.find(hotkey)
        return neuron is not None and neuron.validator_permit and neuron.stake >= min_stake


def _column(source: Any, name: str, length: int, default: Any) -> list[Any]:
    values = getattr(source, name, None)
    if values is None:
        return [default] * length
    try:
        listed = list(values)
    except TypeError:
        return [default] * length
    if len(listed) < length:
        listed.extend([default] * (length - len(listed)))
    return listed[:length]


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _flag(value: Any) -> bool:
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def snapshot_from(
    source: Any,
    *,
    netuid: int | None = None,
    block: int | None = None,
) -> MetagraphSnapshot:
    raw_hotkeys = getattr(source, "hotkeys", None)
    hotkeys = [str(h) for h in raw_hotkeys] if raw_hotkeys is not None else []
    size = len(hotkeys)

    uids = _column(source, "uids", size, 0)
    coldkeys = _column(source, "coldkeys", size, "")
    stake = _column(source, "S", size, 0.0)
    trust = _column(source, "T", size, 0.0)
    vtrust = _column(source, "Tv", size, 0.0)
    incentive = _column(source, "I", size, 0.0)
    emission = _column(source, "E", size, 0.0)
    permits = _column(source, "validator_permit", size, False)
    active = _column(source, "active", size, True)
    last_update = _column(source, "last_update", size, 0)
    axons = _column(source, "axons", size, None)

    neurons = tuple(
        Neuron(
            uid=int(_number(uids[i])),
            hotkey=hotkeys[i],
            coldkey=str(coldkeys[i]),
            stake=_number(stake[i]),
            trust=_number(trust[i]),
            validator_trust=_number(vtrust[i]),
            incentive=_number(incentive[i]),
            emission=_number(emission[i]),
            validator_permit=_flag(permits[i]),
            active=_flag(active[i]),
            last_update=int(_number(last_update[i])),
            address=str(getattr(axons[i], "ip", "") or ""),
            port=int(_number(getattr(axons[i], "port", 0))),
        )
        for i in range(size)
    )

    return MetagraphSnapshot(
        netuid=int(netuid if netuid is not None else _number(getattr(source, "netuid", 0))),
        block=int(block if block is not None else _number(getattr(source, "block", 0))),
        neurons=neurons,
    )


def snapshot_of(netuid: int, block: int, neurons: Sequence[Neuron]) -> MetagraphSnapshot:
    return MetagraphSnapshot(netuid=netuid, block=block, neurons=tuple(neurons))
