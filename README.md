<div align="center">

# Microtensor

### Certified specialist models for real work

Bittensor subnet 576 (testnet)

[Website](https://microtensor.cloud) •
[Docs](https://www.microtensor.cloud/docs/certified-inference) •
[Arenas](https://github.com/microtensor-io/microtensor-arenas) •
[Miner template](https://github.com/microtensor-io/microtensor-miner-template)

</div>

---

## What Microtensor is

Microtensor is a Bittensor subnet that produces certified specialist AI systems: small language models fine tuned for one real task, wrapped in the code that makes them reliable, and proven on hidden tests before anyone can use them. The network pays the people who build the best system for each task and serves the winners as products through the Microtensor API.

## Why

Frontier models answer everything, at frontier prices and frontier latency. Most production work is narrow: compact an agent's context, read a document page, write one database query. A small model trained for that one job runs on a single GPU, a laptop or a phone at a fraction of the cost, once someone has proven it is good enough. Microtensor is where that proof happens.

## What we build

Each task is an **arena**. An arena names one base model, the task, the hidden test items, the grader and the limits a system has to run within. Miners take the base model, make it better at that task, and ship a system in four parts:

| Part | What it does |
|---|---|
| Specialist model | the base model, fine tuned and compressed for the task |
| Harness | the code that drives the model with tools and checks its answers |
| Router | a small classifier that says when the specialist's answer can be trusted |
| Escalation | the repair path that hands a doubtful answer to a frontier model |

The best system in each arena is certified and served as a product.

## How certification works

Every epoch, miners run their system on fresh hidden items on their own GPU and upload a signed transcript of everything it did. Validators replay that transcript on CPU, grade the answers once the labels are released, and set weights. Nothing a miner claims is used; only what the replay confirms counts. The mechanism in full is in the [docs](https://www.microtensor.cloud/docs/certified-inference/system/decentralised-evaluation).

## Arenas

| Arena | Task | Base model |
|---|---|---|
| `ctx_compress` | compact a coding agent's context so the session can continue | Qwen2.5 Coder 7B Instruct |
| `doc_parse` | turn a document page into clean, structured Markdown (draft) | LightOnOCR-2-1B |
| `sql_gen` | answer a plain language question with one correct SQLite query | Qwen2.5 Coder 7B Instruct |

Every arena's definition and page lives in [microtensor-arenas](https://github.com/microtensor-io/microtensor-arenas).

## Products

Certified systems are served through the Microtensor API and console at [microtensor.cloud](https://microtensor.cloud), priced per task, with the certification behind every model in the open.

## Get started

- **Mining**: the [miner guide](https://www.microtensor.cloud/docs/certified-inference/guides/mining), the [training guide](https://www.microtensor.cloud/docs/certified-inference/guides/training), the repository guide [docs/miner.md](docs/miner.md) and the [miner template](https://github.com/microtensor-io/microtensor-miner-template).
- **Validating**: the [validator guide](https://www.microtensor.cloud/docs/certified-inference/guides/validating) and the repository guide [docs/validator_setup.md](docs/validator_setup.md).

## License

Copyright (c) 2026 Microtensor. All rights reserved. This code is licensed for participation in the Microtensor subnet as a miner or validator under the terms at https://www.microtensor.cloud/terms; no other use, redistribution or sublicensing is permitted without written permission from Microtensor.
