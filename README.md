<div align="center">

# Microtensor

**Frontier accuracy at one four-hundredth the size, with the size guaranteed.**

Bittensor subnet 70

[Mine](docs/miner_setup.md) · [Validate](docs/validator_setup.md) · [Mechanism](docs/mechanism.md)

</div>

---

## The problem

A 1.2-trillion-parameter model can do your task. It cannot do your task on a
laptop, in a hospital with no outbound network, on a factory line, or inside a
phone. So the industry ships a compromise: a smaller model with a benchmark
score attached and no commitment whatsoever about what it costs to run.

That benchmark score is the wrong number. A deployer does not need to know a
model scored 71.4 on some public set. They need to know it fits in 3 GB, answers
in under 180 ms at the ninety-fifth percentile, and will still do both under
sustained load at their maximum context. Nobody publishes that, because nobody
is paid to measure it.

## How Microtensor is different

Most inference subnets score behaviour. They send a prompt, wait, and judge the
answer. That measures a *service*, running on hardware the miner chose, under
conditions the miner partly controls. Two miners with identical models and
different GPUs get different scores.

Microtensor scores the **artifact**. Miners submit weights, not an endpoint. The
network holds those weights and runs them itself, on its own certified hardware,
and measures what they cost before it ever asks whether they are any good.

```
measurement  →  gate (binary)  →  accuracy (only among survivors)  →  weights
```

The gate is not a term in a weighted sum. A model that misses the memory ceiling
scores zero no matter how accurate it is. Fit is a precondition for having a
score at all.

What comes out the other end is a **Verified Model Certificate**: a model, and a
signed claim about what it costs to run, produced by a network with no incentive
to flatter it.

## How it works

### Miners ship a file, not a service

There is no axon, no request handler, no GPU that has to stay warm. A miner
compresses a model, hosts the artifact wherever it likes, and commits an 84-byte
pointer on chain.

```
mt1|41|code|laptop|3f9a…c21b|hf:youracct/mt-code-3b@v1
```

Validators fetch that artifact and run it themselves. A miner's box can be
offline between rounds and lose nothing, because the miner's hardware is never
part of the measurement. That is the whole point.

### Nobody can see the questions before they commit

Rounds are windows of blocks, computed from block height, so every validator
agrees on the schedule with no coordination.

```
[start ─────────────────── close) [close ──────── deadline] [·· margin ··]
 submissions open           freeze  evaluate + settle          extrinsic slack
```

The task set is seeded from the hash of the **close** block, the exact block at
which submissions stop. Seeding from the start block, as the obvious design
does, would let anyone read the chain, compute the seed, and then submit with
the questions already in hand. Here the seed cannot exist until it is too late
to use, and it is public immediately after, so anyone can reproduce the draw.

### The envelope is measured, never reported

The miner declares what its artifact costs. The validator measures what it
actually costs, on the reference device for that class, at the declared maximum
input, under sustained load, sampling resident memory throughout and keeping the
maximum. Not the mean, not the value at load.

Both numbers are then checked against each other:

```
measured > class ceiling      → inadmissible
measured > your declaration   → inadmissible
```

Over-declaring is safe but weakens your published certificate, and a deployer
picks the tighter guarantee. Under-declaring is fatal. Honesty is not asked for;
it is the profit-maximising move.

### A miner's score never depends on which validator drew it

Two honest validators emit byte-identical weight vectors. Task selection sorts
by keyed SHA-256 rather than a seeded RNG, so it survives interpreter and
library changes. Scores quantise on a fixed grid, so summing in a different
float order cannot produce a different answer. Any divergence is therefore
attributable rather than excusable as sampling noise.

### Faults are attributed before they are punished

**A fault of the artifact scores zero. A fault of the infrastructure abstains.**

A model that hangs, crashes or exceeds its declaration is the model's problem
and is scored deterministically. A source that is unreachable, or an engine that
will not load, is the *validator's* problem, and it sets no weights at all that
round rather than publishing a partial vector. A missing track is a consensus
divergence, not a smaller sample.

The execution jail fails closed. A host that cannot enforce CPU and memory
limits refuses to run untrusted artifacts, and every result carries whether the
sandbox was genuinely enforced, so an unverified measurement can never reach a
weight vector.

## The competitions

Four tracks, four hardware classes, sixteen independent competitions.

| Track | Metric | Share |
|---|---|---|
| `code` | execution pass rate, schema conformance | 30 % |
| `document` | extraction F1, span accuracy | 30 % |
| `analytics` | exact match, numeric tolerance | 20 % |
| `support` | rubric F1, tool-call correctness | 20 % |

| Class | Size | Peak RSS | p95 TTFT | Reference device |
|---|---|---|---|---|
| `server-cpu` | 8 GiB | 16 GiB | 400 ms | x86-64 server, no accelerator |
| `edge-gpu` | 2.5 GiB | 4 GiB | 120 ms | consumer or embedded GPU |
| `laptop` | 1.5 GiB | 3 GiB | 180 ms | developer workstation |
| `embedded` | 600 MiB | 1 GiB | 300 ms | mobile SoC or NPU |

Each competition pays its top 8 on a geometric curve, so rank 8 still earns
about a third of rank 1 and the tail is worth contesting. Classes rotate across
rounds: the architecture that wins at 8 GB is not the one that wins at 600 MB,
which is what stops a single permanent winner.

Every metric is **computed, not judged**. No model renders an opinion on another
model's output. If a track's quality cannot be reduced to a deterministic
computation against ground truth, it does not become a track.

## Getting started

```bash
mt inspect tracks     # the competitions, their ceilings and emission shares
mt inspect engines    # what this host can execute, and whether the jail is enforced
```

**Mining** takes four commands and no uptime commitment.
Read [docs/miner_setup.md](docs/miner_setup.md).

**Validating** runs a neuron continuously and needs Linux and reference
hardware, but no GPU. Read [docs/validator_setup.md](docs/validator_setup.md).

Prove the machinery works before touching a chain:

```bash
mt validator loopback --rounds 3 --miners 4
```

That stands up a synthetic chain, a seeded corpus and fake miners, then runs
real rounds end to end on your own machine.

Hardware floors for both roles are in [min_compute.yml](min_compute.yml). The
full specification, including everything frozen and everything tunable, is
[docs/mechanism.md](docs/mechanism.md).
