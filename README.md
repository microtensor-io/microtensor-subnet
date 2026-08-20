# Microtensor

**Certified inference systems, measured on the hardware they claim.**
Bittensor subnet 92.

Nobody deploys a model. They deploy a system: a small specialist answers most
queries on cheap hardware, and a routing gate escalates the rest. That
composition decides your real cost and your real quality, and no model card
tells you either number.

Microtensor rewards miners for building those systems and measures what they
actually cost.

1. **Miners** build three components: a **front model** compressed from a
   pinned base to fit a hardware envelope, a **router** that decides when the
   front is out of its depth, and an **escalation specialist** for what the
   front cannot handle. They upload the artifacts and commit a 128-byte
   pointer on chain before the round closes.
2. **Validators** fetch every component and run the assembled system
   themselves: they measure size, sustained peak memory and p95 latency on
   certified reference hardware, gate on the class ceiling and each
   component's own declaration, then score the system end to end against a
   chain-seeded hidden task set and record what it cost per query.
3. **The chain** aggregates validator weights through Yuma Consensus. Systems
   are paid by the ground they uniquely cover on the cost-quality frontier,
   and each component is paid the measured difference it makes.

A miner trains offline and comes online once per round to commit a pointer.
A validator runs Linux on CPU alone, with one certified reference device for
the class it serves.

[Mine](docs/miner_setup.md) · [Validate](docs/validator_setup.md) ·
[Mechanism](docs/mechanism.md)

---

## Live competition

| Track | Front class | Size | Peak RSS | p95 first output |
|---|---|---|---|---|
| `code` | `mt-3g` | 1.5 GiB | 3 GiB | 180 ms |

The front class binds the front model and its router, which are co-resident.
The escalation specialist is measured against the host profile.

Rounds settle daily. Every 30 rounds the frontier is frozen and published as a
version, and that release is what the API and the registry serve.

**Metric:** pass@1 (greedy), executed against hidden tests, scored on the
system's **final** answer. Generated code runs in a sandbox and the score is
the fraction of hidden tests that pass. One generation, decoded greedily at
temperature 0. Computed, never judged.

**Payout:** no ranked list. Systems that are not beaten on both quality and
cost sit on the frontier, and each earns in proportion to the region it alone
covers. A system nobody else is near earns well even if something else is more
accurate; a system sitting on top of an existing one earns almost nothing.

**Next track:** `detect` (vision). The registry in
[`tracks.py`](microtensor/core/tracks.py) carries the full multimodal set as
disabled stubs; tracks open by governance as corpora and reference benches come
online.

## What a system is

| Component | Runs on | Job |
|---|---|---|
| **Front** | every query | Small enough for the class, good enough to resolve most traffic |
| **Router** | every query | A pure function over the front's confidence. Threshold table or small ONNX head. No code executes. |
| **Specialist** | escalated queries only | Catches what the front cannot. Its cost enters yours only in proportion to how often it is needed |

A front alone is a valid system. It will sit at the cheap end of the frontier
and earn there. Adding a router and specialist buys quality at the cost of
escalation, and whether that trade moves you forward is exactly what the
frontier measures.

Two things worth knowing before you build a front model. Its accuracy is not
the whole story: because escalation depends on the router reading the front's
confidence, a front that is slightly less accurate but honest about its
uncertainty beats one that is more accurate and overconfident. And aggressive
quantisation degrades that confidence signal faster than it degrades accuracy,
so the cheapest way to lose a competition is to compress until the front no
longer knows when it is wrong.

## Constraints

| | |
|---|---|
| Base models | Pinned allowlist (Qwen3 / Llama 3.2, 0.6B to 4B) in [`constants.py`](microtensor/core/constants.py). Exact HF revisions, published at corpus freeze. Enforced per component |
| Corpus | Public train split and prompts published per version; hidden tests and the rotating draw are validator-side |
| Reference completions | One vetted completion per train-split task ships with each corpus release, so miners train on prompt/completion pairs without running a large model |
| Role baselines | A baseline front, router and specialist ship with the corpus. Your component is paid the measured difference against the baseline for its role |
| Router forms | JSON threshold table, or ONNX over a published operator allowlist. Features are computed by the validator; the artifact supplies none |
| Training provenance | Public W&B run in `microtensor/training-runs`, bound to each component's digest, required for admission |
| Hardware class | A memory envelope, not a device. `mt-3g` means 3 GiB peak resident memory at your declared maximum input, 1.5 GiB on disk, 180 ms p95 first output |
| Formats | safetensors, GGUF, ONNX |
| Submission | One system per competition per round; chain rate-limits commits to one per 20 minutes |
| Artifact hosting | `hf:` (repo@commit-sha), `ipfs:`, `s3:`, `r2:`, `https:`. All digest-verified after fetch |

## Rounds

A round is 7 200 blocks (about 24 h). Submissions close 600 blocks before the
round ends, and the task set is seeded from the hash of that close block, so
nobody can see the questions while submissions are open and anyone can
reproduce the draw afterwards. Every component must be fetchable by validators
at the close block.

## Getting started

```bash
pip install -e .
mt inspect tracks          # live competitions, ceilings, emission weight
mt inspect engines         # what this host can execute; whether the jail binds
mt inspect readiness       # which launch values are still unset

# Miner
mt miner init --track code --class mt-3g \
  --front ./front --router ./router.json --specialist ./specialist
mt miner selfcheck         # per-component envelopes
mt miner simulate          # resolve rate, expected cost, end-to-end quality
mt miner package           # prints the digests to log to your training runs
mt miner provenance        # confirm each run resolves and binds
mt miner ship              # upload and commit

# Validator
mt validator loopback --rounds 3 --miners 4
mt validator certify mt-3g
mt validator run
```

`mt miner simulate` is the one to spend time in. It runs the whole cascade over
the public train split and reports what fraction resolves at the front, what
the system costs per query, and the end-to-end quality, so router thresholds
can be tuned locally rather than one round at a time.

Hardware guidance for miners and the enforced validator floor are in
[`min_compute.yml`](min_compute.yml). The full mechanism, covering what is
frozen, what is tunable and why every rule exists, is
[`docs/mechanism.md`](docs/mechanism.md).

## Status

Subnet 92 on finney. The mechanism is complete and runs end to end today:
`mt validator loopback` settles rounds on your own machine with no chain and
no network.

Launch values are not yet published. The corpus, reference completions, role
baselines, pinned base-model revisions and the `mt-3g` device profile land
together at corpus freeze. Run `mt inspect readiness` for the current state of
each gate.
