# Running the Microtensor coordinator

This guide is for the team that operates subnet 92. One machine runs the
coordinator; everyone else runs a worker validator. If you are a third party
setting up measurement hardware, read [validator_setup.md](validator_setup.md)
instead. The two audiences share almost nothing.

---

## 1 · What the coordinator is

It derives the round, assigns measurement work, collects signed reports,
reconciles them, computes the canonical settlement, and publishes it. Then it
sets weights with its own stake, and workers that agree submit the same vector.

**It measures nothing.** No model loads on it. No miner artifact is ever
fetched or executed. It reads the chain, serves JSON, stores rows, and computes
a settlement from numbers other people produced.

That last paragraph is a design property worth defending. If a future change
proposes that the coordinator fetch or run an artifact, that change is turning
it into a worker, and the machine should be reclassified rather than quietly
grown.

### Where its authority comes from

Yuma Consensus computes emissions as the stake-weighted median of every
validator's vector, with values above the median clipped. A subnet cannot turn
that off, so the coordinator holds no protocol privilege. Its authority is
ordinary and comes from three things:

- It carries dominant stake, so its vector anchors the median
- Workers submit the published settlement, so the median lands on it rather
  than dispersing across near-identical vectors
- Scoring is deterministic, so a worker that diverges is visible rather than
  merely different

A worker is never asked to trust the coordinator. It recomputes the settlement
from the published reports and refuses to submit on mismatch, which is what
makes the third point real.

---

## 2 · What it needs

| | Coordinator | Worker validator |
|---|---|---|
| CPU | 2 cores | 8 cores |
| RAM | 4 GB | 32 GB |
| Disk | 40 GB SSD | 1 to 2 TB NVMe |
| GPU | none | none |
| Reference devices | none | one per certified class |
| Jail / sandbox | not installed | required, fail-closed |
| Inbound ports | 443 | none |
| Fetches artifacts | never | every round |

A small cloud instance runs it. The settlement is frontier maths over a few
hundred points and a median over a few thousand reports, and finishes in
milliseconds. Size for API concurrency and report retention, not for compute.

---

## 3 · Install

The coordinator extra deliberately excludes the measurement stack: no
`onnxruntime`, no profiler, no jail. A coordinator host that cannot import the
harness is one that cannot be quietly repurposed into a worker.

```bash
git clone https://github.com/microtensor-io/microtensor-subnet
cd microtensor-subnet
python -m venv .venv && source .venv/bin/activate
pip install ".[coordinator]"

mt coordinator init --netuid 92
```

`init` claims the home directory for the coordinator role. From then on
`mt validator run` refuses to start against that home, and `mt coordinator`
refuses to run against a worker home. The check exists because the alternative
is remembering, and the failure mode is a jail on a web server.

```bash
mt coordinator serve --host 0.0.0.0 --port 8443
```

---

## 4 · The endpoints

| | |
|---|---|
| `GET /v1/round/current` | round index, close block, seed, config and its hash |
| `GET /v1/assignment/{hotkey}` | the systems this worker measures |
| `POST /v1/report` | a signed measurement bundle |
| `GET /v1/settlement/{round}` | the canonical weights, once quorum is reached |
| `GET /v1/reports/{round}` | every report the settlement was computed from |
| `GET /v1/frontier/{track}/{class}` | the certified frontier, public |
| `GET /v1/reputation` | per-worker agreement, public |
| `GET /v1/health` | round state, quorum, divergence rate |

Assignments and reports are hotkey-authenticated. The worker signs the method,
path, timestamp and body with its hotkey, and the coordinator verifies against
the metagraph. Covering method and path means a signed report cannot be
replayed against another endpoint. An unregistered hotkey is rejected.

`/v1/reports` is not optional decoration. It is what makes the settlement
reproducible by anyone, and it is the audit trail the first time a miner
disputes a score.

---

## 5 · Assignment

Every system is measured by at least three workers, which is what makes
cross-checking possible at all. The assignment is a keyed SHA-256 shuffle per
system, deterministic in the round seed, the system set and the worker set.

```bash
mt coordinator open
```

`open` reads the round from chain: it derives the round from block height, reads
every commitment, builds the system list, assigns it, and stores the catalogue
of who owns what. The catalogue is the record of which submissions exist this
round and at which uid each miner sits. It is not a task list; tasks stay
derived from the close-block seed against each worker's local corpus, unchanged.

`mt coordinator assign` remains, for replaying an assignment from files:

```bash
mt coordinator assign --round 41 --seed <seed> \n  --systems systems.json --workers workers.json
```

### An unassigned round is not an idle one

Until a round has been assigned, `/v1/assignment/{hotkey}` returns **no
`systems` key at all**, and a worker reads that as a fault rather than as having
nothing to do.

Worth knowing before you debug a quiet fleet. Had the endpoint returned an empty
list instead, every worker would have concluded it was legitimately unassigned,
none would have measured, quorum would never have been reached, and the whole
fleet would have abstained while logging that everything was fine. Absence and
emptiness must not share an encoding across a network boundary.

| response | the worker reads | what to do |
|---|---|---|
| no `systems` key | the round is not assigned yet | run `mt coordinator open` |
| `"systems": []` | assigned nothing this round, ordinary | nothing |
| `"systems": [...]` | measure these | nothing |

A silent fleet with `mt coordinator status` reporting 0 expected reports is the
first row, and `open` is the fix.

**Publish the map.** Anyone holding the seed and the metagraph can recompute it
and check the coordinator assigned honestly. That defence costs one hash, and
an unauditable assignment function is the easiest place for a coordinator to be
accused of favouritism.

Systems that could not reach the replication target are reported rather than
hidden. A system measured by one worker is a system nobody cross-checked, and
the settlement says so.

---

## 6 · Reconciliation

Quality is deterministic. Three workers measuring the same system on the same
task set produce identical figures, so a disagreement is a defect rather than
noise, and the rules follow from that:

| | |
|---|---|
| Quality | exact agreement; the majority stands and outliers are named |
| No majority | the system is unscored for the round, logged loudly |
| Envelope, cost, energy | median across workers whose device profile conforms |
| Version mismatch | the report is rejected, not reconciled |

**Divergent quality is never averaged.** Averaging would turn a defect into a
plausible number and hide it.

**A three-way split is not broken by the coordinator.** If all three workers
differ, the system goes unscored. A coordinator that picks a winner among three
disagreeing measurements is deciding the outcome rather than counting it, and
that is the exact power this design is trying not to have.

**A version mismatch is a rejection, not a divergence.** A worker on a
different engine or corpus version is misconfigured, and folding its numbers
into the majority is how a subtly wrong answer becomes canonical.

---

## 7 · Config anchoring

The coordinator serves the competition config: live tracks, ceilings, emission
weights, corpus version, role baselines. Serving it is fine. Serving it
unanchored is not, because the rules could then change without anyone being
able to prove they had.

At round start, commit the config hash on chain:

```bash
mt coordinator config          # prints the config and its hash
```

Workers verify the served config against that anchor and abort the round on
mismatch. Note that they abort rather than falling back: measuring against
ceilings nobody committed to produces numbers that look valid and are not.

An empty anchor is a mismatch, not a pass. An unverifiable config is precisely
the state the check exists to refuse.

---

## 8 · When the coordinator is down

Workers retry with backoff inside the round window. If it is still unreachable
at the deadline they fall back to standalone: derive the round from chain
state, measure what they can, settle independently, and submit their own
vector. The fallback is logged prominently.

Standalone rounds are slower and cover fewer systems, but the subnet continues
and weights are still set. Without this a coordinator outage would halt the
network and nobody would earn.

Run the fallback deliberately at least once before you depend on it:

```bash
mt validator run --coordinator http://127.0.0.1:1 --max-rounds 1
```

---

## 9 · Operating it

**The hotkey.** It signs every settlement, which makes it the most sensitive
credential in the system. Put it in a secrets manager, not an environment file.

**Backups.** Reports are the audit trail and cannot be reconstructed. Back the
database up on a schedule and test a restore. Retention: reports and
settlements indefinitely, assignments prunable.

**TLS and limits.** Terminate TLS in front. Rate-limit, and cap request bodies
on `/v1/report`: reports are signed bundles of numbers, so anything large is
malformed by definition.

**Watch.** Quorum reached per round, worker count, divergence rate, reputation
distribution, settlement publish latency. A rising divergence rate usually
means a release went out that changed scoring, and it is the first place that
shows.

```bash
mt coordinator status --round 41
```

**Restart policy** is `on-failure`, not `always`, matching the fail-closed
posture everywhere else. A coordinator that exits deliberately, on a config
anchor mismatch or a database failure, should stay down and page someone rather
than loop.

**One instance.** If it later needs redundancy, the binding constraint is that
only one node may sign a settlement for a given round. Run a hot standby with
manual promotion; two signed settlements for one round is the worst failure
this system can produce.

---

## 10 · What this costs, honestly

The coordinator is a service you must keep up, secure and scale, and its API is
an attack surface. A compromised coordinator publishing a false settlement is
caught only by workers actually recomputing it, which is why that step is in
the worker and not left to good intentions.

The decentralisation claim changes shape. It is no longer "no coordinator". It
is "a coordinator whose every input, assignment and output is published and
independently recomputable". That is a defensible claim and a materially
stronger one than serving opaque scores from a benchmark API, but it is a
different claim, and the whitepaper states the new one rather than the old.
