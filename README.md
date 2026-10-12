<div align="center">

# Microtensor

### A Decentralized Network for Certified Inference Systems

Bittensor subnet 576 (testnet)

[Mining](docs/miner_setup.md) •
[Inference miner](docs/inference_miner.md) •
[Compute miner](docs/compute_miner.md) •
[Validator](docs/validator_setup.md) •
[Mechanism](docs/mechanism.md) •
[Dashboard](https://microtensor.cloud)

</div>

---

## What the network does

Microtensor pays for specialised small language models and the machinery around them. A miner ships a system in four parts: a small model trained for one task, a harness that drives it with tools, a router that decides when the small model's answer can be trusted, and an escalation path that repairs a wrong answer with a frontier model. Validators run every part themselves on their own GPUs, against hidden items they decrypt only once an epoch opens, and score each part in its own pool.

Nothing a miner claims is used. The artifact is fetched from Hugging Face at the revision committed on chain and executed inside a gVisor sandbox that reaches only the validator's own services.

## How an epoch works

1. **Commit.** A miner pays the arena fee and commits one line on chain: arena, Hugging Face user, artifact commit, fee block and extrinsic index.
2. **Roster and snapshot.** At commit close the server verifies every fee and publishes a signed roster and a signed snapshot of previous bests.
3. **Release.** Once the open block is final the server seals the epoch's encrypted item bundle to every permitted validator and publishes the seed, the hash of the open block hash and the arena id.
4. **Evaluation.** Each validator serves the miner's model with vLLM, serves the miner's router, runs the miner's `main.py` in the sandbox, and runs the SLM, harness, router, escalation and system tests.
5. **Report.** Each validator signs one report per epoch and sets weights from its own scores.
6. **Ledger and archive.** The server publishes the stake weighted median across agreeing reports, applies the ladder, and mirrors the SLM pool winner privately to the archive.

Five pools pay 20% each by rank (40 / 25 / 15 / 12 / 8). Every score is a one sided lower bound; a published score only moves when a new result beats the previous best for the same artifact by 0.01.

## Repository map

| Path | What it holds |
|---|---|
| `microtensor/chain` | commitments (`mt2`), epochs, weights, chain client |
| `microtensor/core/items.py` | item bundles, encryption, sealed keys, release manifests |
| `microtensor/artifact` | the manifest contract, the allowed package list, validation |
| `microtensor/validator` | the epoch pipeline: collect, sandbox, tests, scoring, report |
| `microtensor/scoring` | lower bounds, pools, SQL execution grading |
| `microtensor/cli/submit.py` | `mt submit manifest`, `validate`, `pay`, `commit`, `status` |
| `microtensor/cli/epoch.py` | `mt epoch register-key`, `relay-token`, `run`, `loop` |
| `microtensor/serving`, `microtensor/rigs` | the certified inference serving product and the compute pool |
| `deploy` | the validator and sandbox images and the validator compose file |

## Documentation

The full documentation is on the site: https://www.microtensor.cloud/docs/certified-inference. The miner template is at https://github.com/microtensor-io/microtensor-miner-template.

## License

Copyright (c) 2026 Microtensor. All rights reserved. This code is licensed for participation in the Microtensor subnet as a miner or validator under the terms at https://www.microtensor.cloud/terms; no other use, redistribution or sublicensing is permitted without written permission from Microtensor.
