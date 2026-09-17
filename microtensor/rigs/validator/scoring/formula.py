from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# score = base × w_d × s_c × m_iso × m_up × m_drv × m_opt
# emission = pool × score / Σ score, with pool = 60% of compute emission.
# Every input is a validator observation or a number from the assignment ledger.
# Nothing here reads a value the rig reported about itself.

# Sixty percent of compute emission is distributed by score. The other forty
# percent is held before anything is paid: burn, validator share and reserve.
# A pool paying out everything has nothing to adjust with when demand shifts.
POOL_SHARE = 0.60

# A machine without proper container isolation is still useful for inference
# and mining; it cannot be trusted with a stranger's code, so it earns less.
# A cost rather than a gate.
ISOLATION_MULTIPLIER = 0.80

# Uptime ramps linearly to one over fourteen days. Probation expressed as
# money: churn is unprofitable and a proven machine is worth more than an
# identical unproven one.
UPTIME_RAMP_DAYS = 14.0

# Rental sits first in the priority ladder. A machine declining it is
# unavailable for the most valuable work, so the floor must be lower or
# everyone would decline and rental would have no supply.
OPT_OUT_MULTIPLIER = 0.85


@dataclass(frozen=True)
class ScoreInputs:
    # base: the deep pass verdict for this epoch. Failed verification scores
    # zero and earns nothing from either stream.
    verified: bool
    # w_d: demand weight, the exponential moving average of revenue per GPU
    # type. See demand_weight.py. GPU types earning more get a larger share.
    demand_weight: float
    # s_c: capacity share, the machine's GPUs over the fleet's verified GPUs.
    # Twenty cards earn twenty times one card, which keeps the per GPU entry
    # fee proportionate.
    gpu_count: int
    fleet_gpu_count: int
    # m_iso: whether container isolation (sysbox with ID mapped mounts) works.
    isolation_ok: bool
    # m_up: days of accumulated uptime from the reliability ledger.
    uptime_days: float
    # m_drv: the driver at or above the announced minimum; the cutoff has
    # passed; and whether a customer currently holds the machine.
    driver_ok: bool
    driver_cutoff_passed: bool
    in_rental: bool
    # m_opt: the owner declined rental.
    rental_opted_out: bool


@dataclass(frozen=True)
class ScoreBreakdown:
    base: float
    demand_weight: float
    capacity_share: float
    isolation: float
    uptime: float
    driver: float
    opt_out: float

    @property
    def score(self) -> float:
        return (
            self.base
            * self.demand_weight
            * self.capacity_share
            * self.isolation
            * self.uptime
            * self.driver
            * self.opt_out
        )

    def payload(self) -> dict[str, float]:
        return {
            "base": self.base,
            "w_d": self.demand_weight,
            "s_c": self.capacity_share,
            "m_iso": self.isolation,
            "m_up": self.uptime,
            "m_drv": self.driver,
            "m_opt": self.opt_out,
            "score": self.score,
        }


def uptime_multiplier(uptime_days: float) -> float:
    # Linear ramp, clamped: zero uptime pays nothing extra, fourteen days pays
    # in full. Uptime is a multiplier on verified work, never a term of its own,
    # so a machine that is online and does nothing still scores zero through
    # base.
    return min(1.0, max(0.0, uptime_days) / UPTIME_RAMP_DAYS)


def driver_multiplier(driver_ok: bool, cutoff_passed: bool, in_rental: bool) -> float:
    # A hard gate, phased. The driver runs privileged and a vulnerability there
    # reaches every tenant on the host, so an out of date driver eventually
    # earns nothing. The cutoff is announced with a grace period, because a
    # contributor who has not read the announcement is not an attacker. A
    # machine mid rental is exempt entirely: a customer must never be
    # interrupted because their host has not upgraded.
    if driver_ok or in_rental:
        return 1.0
    return 0.0 if cutoff_passed else 1.0


def breakdown(inputs: ScoreInputs) -> ScoreBreakdown:
    # base comes only from the deep pass. Not from the express tick, not from
    # the websocket, not from anything the agent said.
    base = 1.0 if inputs.verified else 0.0
    # The demand weight is never allowed below zero; a tier with no revenue yet
    # carries a small static prior instead (demand_weight.py), so a new tier is
    # not priced at nothing before the market has spoken.
    demand = max(0.0, inputs.demand_weight)
    # Capacity share is the machine's verified GPUs over the fleet's verified
    # GPUs this epoch. A fleet of zero means nobody was verified; the share is
    # zero rather than undefined.
    capacity = (
        inputs.gpu_count / inputs.fleet_gpu_count
        if inputs.fleet_gpu_count > 0 and inputs.gpu_count > 0
        else 0.0
    )
    return ScoreBreakdown(
        base=base,
        demand_weight=demand,
        capacity_share=capacity,
        isolation=1.0 if inputs.isolation_ok else ISOLATION_MULTIPLIER,
        uptime=uptime_multiplier(inputs.uptime_days),
        driver=driver_multiplier(inputs.driver_ok, inputs.driver_cutoff_passed, inputs.in_rental),
        opt_out=OPT_OUT_MULTIPLIER if inputs.rental_opted_out else 1.0,
    )


def score(inputs: ScoreInputs) -> float:
    return breakdown(inputs).score


def emissions(scores: Mapping[str, float], compute_emission: float) -> dict[str, float]:
    # pool × score / Σ score. The forty percent held is not in the pool and is
    # never distributed here; weights.py routes it to the reserve on chain.
    pool = POOL_SHARE * max(0.0, compute_emission)
    total = sum(value for value in scores.values() if value > 0.0)
    if total <= 0.0:
        return {key: 0.0 for key in scores}
    return {key: (pool * value / total if value > 0.0 else 0.0) for key, value in scores.items()}
