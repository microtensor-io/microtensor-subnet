<div align="center">

# Microtensor

### A Decentralized Network for Certified Inference Systems

Bittensor subnet 92

[Miner](docs/miner_setup.md) •
[Validator](docs/validator_setup.md) •
[Coordinator](docs/coordinator_setup.md) •
[Mechanism](docs/mechanism.md) •
[Dashboard](https://microtensor.cloud)

</div>

---

## Introduction

Frontier-quality inference is unavailable in most of the places that need it. At
sustained volume the cost exceeds the value of the work; in interactive systems
the latency exceeds the budget; and in regulated environments the data cannot
leave the network at all, so no remote endpoint is admissible at any price.

Composed systems answer this. The dominant pattern is the cascade: a compact
specialist resident on inexpensive hardware answers the queries it can, a
confidence gate decides, and the remainder escalate to a larger model. Routed
compositions of this kind match frontier answer quality while reducing cost by
more than an order of magnitude, and the position that compound systems are the
unit at which capability is now delivered is well established in the systems
literature.

What such a system costs and delivers, though, is knowable only by measuring it.
Expected cost per query is

```
C  =  c_front  +  (1 - ρ) · c_specialist
```

where `ρ` is the fraction of traffic the front resolves. That fraction is a
joint property of the front model, the routing thresholds and the task
distribution, so no component's published figures determine it. End-to-end
quality likewise depends on what the system finally returns rather than on what
the front returns alone, and resident memory under production context routinely
exceeds what a parameter count implies, because the cache term grows with input
length rather than with weight count.

Each of these emerges from the assembled composition running on specific
hardware.

Established evaluation ranks single frozen models against a quality metric. That
measures a well-defined property and is the right instrument for comparing
models. Microtensor supplies the complementary instrument: it measures assembled
systems on the hardware they will run on, and publishes each result as a signed
certificate bound to the exact artifacts that produced it.

---

## How it works

```
                                    ┌──────────────┐
                        escalate    │  specialist  │
                     ┌─────────────▶│ host profile │───┐
                     │   (1 - ρ)    └──────────────┘   │
  ┌───────┐   ┌──────┴──────┐                          ▼
  │ query │──▶│    front    │──▶ ◇ router          ┌────────┐
  └───────┘   │ class limit │                      │ answer │
              └─────────────┘──────────────────────└────────┘
                                    resolve  (ρ)
```

The front runs on every query, so its envelope determines the economics of the
whole system. The specialist runs only on the escalated fraction, so its cost
enters the total weighted by `1 - ρ`.

Microtensor rewards miners for building these systems and measures what they
cost. The mechanism works as follows:

1. **Miners** build up to three components: a **front model** compressed from a
   pinned base to fit a hardware envelope, a **router** that decides when the
   front is out of its depth, and an **escalation specialist** for what the
   front cannot handle. They upload the artifacts and commit a pointer on chain
   before the round closes.

2. **Validators** fetch every component and run the assembled system themselves.
   They measure size, sustained peak memory and p95 latency on certified
   reference hardware, gate on both the class ceiling and each component's own
   declared envelope, then score the system end to end against a hidden task set
   seeded from the block at which submissions closed.

3. **The chain** aggregates validator weights through Yuma Consensus. Systems
   earn in proportion to the region they uniquely cover on the cost-quality
   frontier, and each component earns the measured difference it makes against a
   published baseline for its role.

Miners train offline and come online once per round to commit a pointer.
Validators run on commodity hardware with one certified reference device per
class they serve.

See the [Miner](docs/miner_setup.md) and [Validator](docs/validator_setup.md)
docs for how each role works and how to set one up.

---

## Incentive mechanism

Each competition pairs a task track with a hardware class, and the class is a
memory envelope rather than a device. A system is admissible only when every
component fits both the class ceiling and the envelope its author declared for
it, so a certificate reports figures that are binding rather than aspirational.

Admission is binary. A system exists for a round only when every component
clears both bounds on every axis:

```
admissible  ⇔  measured ≤ declared ≤ class ceiling
                for size, memory, latency, on every component
```

Among admissible systems, reward follows position on the cost-quality frontier.

```
  quality
     ▲
     │                              ● C
     │                     ┌────────┘
     │            ● B      │ region B
     │       ┌─────────────┘ alone covers
     │  ● A  │           ·        · ← dominated systems
     │       │      ·
     └───────┴────────────────────────────────▶ cost
```

A system that is not beaten on both axes sits on the frontier and earns in
proportion to the region it alone covers:

```
E(S)  ∝  HV(frontier)  −  HV(frontier without S)
```

A system placed beside an existing one covers almost nothing the other does not,
and both earn less as a result. A system opening an unoccupied trade-off earns in
proportion to the ground it opens. There is no ranked list and no weighted
composite score, so no arbitrary exchange rate between quality and cost is
imposed on participants.

Within a system, each component is settled by role-baseline ablation: the
measured change in the system's value when that component is replaced by the
published baseline for its role.

```
φ(component)  =  V(system)  −  V(system with component → role baseline)
```

A component the system does not need measures zero and earns nothing, and a
router that improves the system's frontier position by trading escalation
against quality is paid exactly the value of that improvement.

Each class is a standing competition that opens and remains open, so a new class
is an addition rather than a replacement. The target stays in motion through
reference-model advance, corpus rotation, the combinatorial growth of the
composition space, and incumbent decay, each of which keeps the frontier moving
while preserving work already completed for an existing envelope.

---

## Why measurement is the product

Three properties distinguish a Microtensor certificate from a reported benchmark
score.

**The network holds the weights and runs them.** A miner submits artifacts and
is absent at scoring time, supplying no measurements of its own. Because the
network controls execution, every figure is generated by the measuring party
rather than by the party being measured, and the system that earns a certificate
is byte for byte the system a deployer receives.

**Scoring is deterministic.** Generation decodes greedily at temperature zero
under a pinned runtime, and the router is required to be a pure function of
deterministic features. Independent validators evaluating the same system
therefore obtain identical results, which makes disagreement between them a
defect that can be attributed rather than variance that must be tolerated.

**Evaluation is unobservable in advance.** The task set is seeded from the hash
of the block at which submissions close, so the draw cannot be observed while
submissions are open and can be reproduced by any party immediately afterwards.
A rotating corpus partition defeats memorisation, and a fixed partition held
constant across rounds makes improvement over time a claim about capability
rather than an artefact of draw difficulty.

The full specification, including what is frozen and what is tunable, is in
[`docs/mechanism.md`](docs/mechanism.md).

---

## Rounds and releases

A round settles daily. Every 30 rounds the frontier is frozen and published as a
signed, immutable version, and that release is what interfaces serve until the
next supersedes it, so a deployment can pin a version rather than track a
surface that moves each day.

Live competitions, their ceilings and current results are published on the
[dashboard](https://microtensor.cloud). `mt inspect tracks` reports the same
from a local checkout.

---

## Status

Subnet 92 on finney. The mechanism is complete and runs end to end:
`mt validator loopback` settles rounds against a synthetic chain on a local
machine, with no wallet and no network.

Launch values are published at corpus freeze. Run `mt inspect readiness` for the
current state of each gate.

---

## License

MIT
