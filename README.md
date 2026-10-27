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

Nothing a miner claims is used. The artifact is fetched from Hugging Face at the revision committed on chain and executed inside a locked down Docker container that reaches only the validator's own services: an internal network to the gateway and nothing else, CPU, memory and process limits, read only mounts and every capability dropped.

## How an epoch works

1. **Commit, sealed.** A miner pays the arena fee, if the arena charges one, and commits one line on chain: arena, a digest of (Hugging Face user, artifact commit, secret salt), fee block and extrinsic index (0 and 0 for a free arena). The repositories stay private, so nothing can be copied during the window. `mt submit commit --open` still publishes the repository in the clear.
2. **Reveal.** When the commit window closes a reveal window opens. The miner makes both repositories public and runs `mt submit reveal`, which sends the user, commit and salt signed by the hotkey. The server checks the digest; a missing or mismatched reveal is rejected, and validators recheck the digest themselves from the roster. A model or artifact whose weights duplicate an earlier commitment is rejected as a copy.
3. **Roster and snapshot.** At commit close the server verifies every fee it charges and publishes a signed roster and a signed snapshot of previous bests.
4. **Release.** Once the open block is final the server seals the epoch's encrypted item bundle to every permitted validator and publishes the seed, the hash of the open block hash and the arena id.
5. **Evaluation.** Each validator serves the miner's model with vLLM, serves the miner's router, runs the miner's `main.py` in the sandbox, and runs the SLM, harness, router, escalation and system tests.
6. **Report.** Each validator signs one report per epoch and sets weights from its own scores.
7. **Ledger and archive.** The server publishes the stake weighted median across agreeing reports, applies the ladder, and mirrors the SLM pool winner privately to the archive.

Five pools pay 20% each by rank (40 / 25 / 15 / 12 / 8). Every score is a one sided lower bound; a published score only moves when a new result beats the previous best for the same artifact by 0.01.

## Repository map

| Path | What it holds |
|---|---|
| `microtensor_subnet/chain` | commitments (`mt2`), epochs, weights, chain client |
| `microtensor_subnet/core/items.py` | item bundles, encryption, sealed keys, release manifests |
| `microtensor_subnet/artifact` | the manifest contract, the allowed package list, validation |
| `microtensor_subnet/validator` | the epoch pipeline: collect, sandbox, tests, scoring, report |
| `microtensor_subnet/scoring` | lower bounds, pools, SQL execution grading |
| `microtensor_subnet/cli/submit.py` | `mt submit manifest`, `validate`, `pay`, `commit`, `status` |
| `microtensor_subnet/cli/epoch.py` | `mt epoch register-key`, `relay-token`, `run`, `loop` |
| `microtensor_subnet/serving`, `microtensor_subnet/rigs` | the certified inference serving product and the compute pool |
| `deploy` | the validator and sandbox images and the validator compose file |

## Documentation

The full documentation is on the site: https://www.microtensor.cloud/docs/certified-inference. The miner template is at https://github.com/microtensor-io/microtensor-miner-template.

## License

Copyright (c) 2026 Microtensor. All rights reserved. This code is licensed for participation in the Microtensor subnet as a miner or validator under the terms at https://www.microtensor.cloud/terms; no other use, redistribution or sublicensing is permitted without written permission from Microtensor.
