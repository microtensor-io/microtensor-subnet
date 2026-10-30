<h1 align="center">Microtensor</h1>

<p align="center"><em>Task specialised small language models, trained by a network and proven under adversarial evaluation.</em></p>

<div align="center">

[![Bittensor](https://img.shields.io/badge/Bittensor-Subnet-blue)](https://bittensor.com)
[![Status: Testnet](https://img.shields.io/badge/Status-Testnet%20576-orange)](https://microtensor.cloud/network)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-lightgrey)](https://www.microtensor.cloud/terms)

</div>

<p align="center">
  <a href="https://microtensor.cloud">Homepage</a> &bull;
  <a href="https://www.microtensor.cloud/docs/certified-inference">Docs</a> &bull;
  <a href="https://www.microtensor.cloud/docs/certified-inference/system/decentralised-evaluation">Mechanism</a>
</p>

Microtensor runs on Bittensor testnet as **netuid 576**.

Microtensor is a decentralised network for producing **specialists**: small language models fine tuned from one fixed base for one narrow task, shipped with the harness, router and escalation code that make them usable, and evaluated on hidden items under conditions no participant controls. Every published score is a verified, reproducible measurement, not a self report.

## The measurement gap

Most production language model work is narrow and repetitive: compacting an agent's context, reading a document page, turning a question into a database query. Frontier models solve these tasks at frontier cost and latency. Small models can match them on a narrow distribution, and the literature on task specific fine tuning says so consistently, yet the numbers that would let anyone rely on a small model are the weakest part of the field.

Public benchmarks leak into training data. Self reported evaluations run on the author's hardware, with the author's prompt, on items the author chose. Nothing distinguishes a model that generalises from one that memorised the test, and nothing ties a published number to the weights that produced it. A buyer who needs a specialist has no way to tell which one is real.

## Evaluation as the primary artifact

Microtensor treats evaluation as the product. Each task is an **arena** with one base model, a hidden item pool and a grader, and an epoch is a measurement under fixed conditions:

- **Fresh hidden items every epoch**, drawn by a seed that does not exist until the epoch opens, so no model can have trained on them.
- **One base model per arena.** Every miner fine tunes the same model, so the comparison measures training and compression skill, not the choice of a bigger starting point.
- **The miner runs, the network verifies.** A miner evaluates its own system on its own GPU inside pinned images and uploads a signed transcript. Validators replay the code against that transcript on CPU, and checkers re-run sampled model calls on the same hardware class and require identical tokens.
- **Scores are lower confidence bounds**, paired against the base on the same items, so a lucky run cannot outrank a consistent one.
- **Consensus is on chain.** Validators sign independent reports and set weights; the published figure is the stake weighted median of reports that agree on the inputs.

| | Public benchmarks | Vendor evaluations | Microtensor |
|---|---|---|---|
| Test exposure | public, contaminated over time | private, unverifiable | hidden, replaced every epoch, canaries detect leakage |
| Who runs the model | the author | the vendor | the miner, then replayed and spot checked by others |
| Starting point | any | any | one base model per task |
| Reported figure | point estimate | point estimate | one sided lower bound |
| Reproducibility | rarely | none | deterministic replay of a signed transcript |
| Incentive to be right | none | commercial | TAO emissions against stake weighted consensus |

## Anatomy of an epoch

### Fixed conditions

An arena fixes the task, the base model and its licence, the grader, the tools, the limits a system must run within and the number of hidden items per pool. The arena definition is frozen while it is live; a stronger base model is a new arena version. Epochs are measured in blocks: a commit window in which miners declare their revision and hardware profile, a window in which they run, and a grading margin in which validators report.

### What is measured

A submission is a system of up to four parts: the specialist model, a harness that drives it with the arena's tools and checks its answers, a router that predicts whether the specialist's answer can be trusted, and an escalation path that repairs a doubtful answer through a frontier model. Each part is scored in its own pool, and a missing part is replaced by the arena's reference so the others can still be measured.

### Execution on declared hardware

During the window the miner's frozen artifact and weights are pulled from a content addressed mirror and run inside official images pinned by digest: a deterministic inference engine, a sandbox with a frozen clock and no network, and a gateway that records every model, tool and relay call. The transcript is signed by the miner and sealed under a Merkle root at the close of the window.

### Independent replay and re-execution

Validators need no GPU. They replay the artifact's code on CPU with every model call answered from the transcript, every tool response recomputed and every relay receipt checked, and reject any request that differs from the recording. Checkers holding a GPU of the miner's declared profile re-run at least 200 sampled calls on the same engine and require an exact token match, time the calls against the base model on the same card, and publish any mismatch as evidence that others confirm before it counts.

### From answers to weights

After the labels release, every answer is graded. The model pool score is the lower bound of the pass rate paired against the base on the same items, clustered by item family, multiplied by an efficiency term from the measured latency and memory. Twin items detect memorisation; canaries planted in earlier epochs detect training on released inputs. Validators sign one report per epoch and set weights; the server folds reports into a stake weighted median for publication and never decides a score itself.

## The chain of evidence

```text
arena:        one task, one base model, one hidden pool per epoch
submission:   { hotkey, artifact revision, weights sha256, hardware profile }
transcript:   { per call: prompt sha256, output token ids, tool and relay receipts },
              signed by the miner, sealed under the epoch's receipt root
verification: { replay: ok | first mismatch,
                spot check: calls matched, TTFT, TPOT, peak memory }
scores:       per pool { lower bound, point, items, eligible }
report:       signed by the validator; published figure = stake weighted median
```

The miner signs what its system did; the validator signs what the replay confirmed; the chain carries the consensus. Anyone can verify each link offline.

## Join the network

```bash
git clone https://github.com/microtensor-io/microtensor-subnet.git
cd microtensor-subnet
pip install -e .
```

- **[Miner guide](https://www.microtensor.cloud/docs/certified-inference/guides/mining):** register your hotkey on netuid 576, build your system from the arena's base model, commit it and run your evaluation. Repository guide: [docs/miner.md](docs/miner.md). Start from the [miner template](https://github.com/microtensor-io/microtensor-miner-template).
- **[Validator guide](https://www.microtensor.cloud/docs/certified-inference/guides/validating):** one CPU box, a registered key, the loop. Repository guide: [docs/validator_setup.md](docs/validator_setup.md).

Full documentation lives at [microtensor.cloud/docs](https://www.microtensor.cloud/docs/certified-inference).

## License

Copyright (c) 2026 Microtensor. All rights reserved. This code is licensed for participation in the Microtensor subnet as a miner or validator under the terms at https://www.microtensor.cloud/terms; no other use, redistribution or sublicensing is permitted without written permission from Microtensor.
