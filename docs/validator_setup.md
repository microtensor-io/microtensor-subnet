# Running a Microtensor validator

Netuid **576** on Bittensor testnet.

## Requirements

| | |
|---|---|
| Machine | one CPU box: 8 cores x86-64, 32 GB RAM, 200 GB NVMe |
| OS | Ubuntu 22.04 or 24.04 |
| Software | Docker 27 or newer |
| Network | a stable connection; every epoch the loop downloads each miner's transcript and the epoch bundles |
| Uptime | the loop must be running at each epoch's Freeze 2 and through the 6 hour grading window that follows |
| Access | a hotkey registered on netuid 576, a registered X25519 key, relay access granted by the team |

## 1. Host preparation

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
sudo mkdir -p /var/lib/microtensor && sudo chown 10001:10001 /var/lib/microtensor
```

`/var/lib/microtensor` has to exist at that exact path on the host: the validator bind-mounts it at the same path inside so the replayer containers it starts see the same files.

## 2. Register the hotkey

```bash
btcli subnet register --netuid 576 --subtensor.network test --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

## 3. Keys and access

Register the key, then ask the team to grant relay access to your hotkey.

```bash
docker run --rm -it \
  -v /var/lib/microtensor:/var/lib/microtensor \
  -v ~/.bittensor/wallets:/wallets:ro \
  -e MT_WALLET_PATH=/wallets \
  --entrypoint mt ghcr.io/microtensor-io/validator:1 \
  epoch register-key --wallet.name <coldkey> --wallet.hotkey <hotkey> --home /var/lib/microtensor
```

Once access is granted:

```bash
docker run --rm -it \
  -v /var/lib/microtensor:/var/lib/microtensor \
  -v ~/.bittensor/wallets:/wallets:ro \
  -e MT_WALLET_PATH=/wallets \
  --entrypoint mt ghcr.io/microtensor-io/validator:1 \
  epoch relay-token --wallet.name <coldkey> --wallet.hotkey <hotkey> --home /var/lib/microtensor
```

The key lives at `/var/lib/microtensor/keys/x25519.key` and the relay token beside it. Back the directory up; losing the key means re-registering and waiting for the next epoch's labels release.

## 4. Run

```bash
git clone https://github.com/microtensor-io/microtensor-subnet
cd microtensor-subnet/deploy
cp .env.example .env   # wallet names, DOCKER_GID
docker compose -f docker-compose.validator.yml up -d
docker compose -f docker-compose.validator.yml logs -f
```

The loop takes every live arena from the server itself. At each epoch's Freeze 2 it runs the seven steps below once; an epoch already reported is not run again, and the record is `/var/lib/microtensor/state/<arena>.epochs.json`.

## The seven steps of an epoch

1. **Load the epoch.** The arena definition, the signed `mirror`, `inputs` and `receipts` documents, the seed check against the chain's open block hash, the roster. When the labels bundle carries no reference answers, wait for the coordinator's, with the same time limits as the spot check wait below, and grade its system answers with your own probe model.
2. **Check receipts.** Fetch every miner's transcript from the mirror by its sha256; its leaf must sit under the `receipts` Merkle root. A miner with no upload is unscored.
3. **Replay.** For every miner and every item, run the frozen `main.py` from the mirror in the replayer image with no network and a frozen clock, answer each model call from the transcript, recompute every tool response from the bundle, verify every relay receipt, and require every request to match the recording exactly and in order. The task is rebuilt from the bundle after its `task_sha256` matches. Count calls and tokens per condition; time the harness on your own clock. Any mismatch fails the miner for the epoch. Minutes, no model inference. A miner that passes has its weights pulled from the mirror and its provenance checked against the base its `training/recipe.json` names. A model with no recipe in the mirror is checked against the arena's one allowed base. The change sketch for the duplicate check is built here, about 2 minutes per miner on CPU, before the weights are removed.
4. **Grade.** Grade every answer with the arena grader, using the labels bundle your key opened when the epoch loaded; the probe grader runs the arena's pinned small GGUF model with llama.cpp.
5. **Apply spot checks.** Wait for the coordinator's signed `spotcheck` document, then apply its measured figures. A miner with no spot check result scores zero in the specialist model (`slm`) and system pools for this epoch. A miner with a standing flag in this epoch, or in the earlier epochs of the arena's ban window, scores zero. An appeal the coordinator upheld lifts the flag.
6. **Compare for duplicates.** Fetch each router's weights from the mirror and check its size: a router over the arena's parameter limit scores zero in the system pool. Then compare every submission with all earlier ones in commit order, as below. A duplicated model scores zero in the specialist model and system pools, a duplicated router scores zero in the system pool, and the miner's other pool is still scored.
7. **Score, set weights, report.** The scoring formulas, gates and commit reveal. Two pools are paid. The specialist model pool (`slm`) takes 60% of all miner emission and the system pool takes 40%. Within each pool the top five eligible miners get 40%, 25%, 15%, 12% and 8%, renormalised over the eligible entries, and an empty pool burns to the subnet owner hotkey. The system pool scores the full pipeline: the miner's model, harness, router and escalation working together, with the reference parts wherever the miner ships none. Harness, router and escalation are not paid as pools of their own, and serving and compute take no share of these weights. A chain error while setting weights is logged and the report still goes out. The report carries the receipt root, the pool shares and, for each miner, the transcript sha256, the replay result, the spot check or the reason there was none, the duplicated pools and the canary counts; the server folds reports into the stake weighted median. Between settlements the validator sends the same weights again once in every chain epoch, from a background thread, so it stays active on chain however long an arena epoch or a grading run lasts. Weights the chain refused are kept and sent again a few minutes later. An arena closes when its epoch settles, and the weights from that settlement stay on chain while it is closed. When a new round goes live, of this arena or another, the validator replaces them with a full burn to the subnet owner hotkey until that round settles, so nobody is paid for an old result while a new one is being decided. An epoch with no coordinator report (below) sets no new weights: the validator sends the last settled weights again instead, and the server settles that epoch void. `state/weights.json` keeps the last settled ranking under `settled`, apart from the current weights, so the burn at the start of a round cannot erase it.

## Dry run

```bash
docker compose -f docker-compose.validator.yml run --rm validator epoch run --arena ctx_compress --dry-run --item-limit 20 --home /var/lib/microtensor
```

Replays and grades twenty items per pool, posts nothing and sets no weights. Use it after installation and after every image update.

## Updating

```bash
docker compose pull && docker compose -f docker-compose.validator.yml up -d
```

The validator image is built from `main` and tagged `ghcr.io/microtensor-io/validator:1`. The replayer and llama.cpp images are pulled by the validator itself, by the digests pinned in each arena's definition, never by tag alone.

## Spot checks

Validators need no GPU and run no spot checks. The coordinator is the only part of the evaluation that needs a GPU beyond each miner's own job. The subnet owner runs a coordinator, which re-runs a sample of each miner's model calls on the mirrored weights and requires an exact token match. It restarts its engine and reruns anything that differs, and a call only becomes evidence when both runs agree with each other and differ from the miner. Speed and memory are only measured on the profile card, such as an RTX 4090 for `P1`.

The coordinator signs a report for each miner, and the server publishes them as the epoch's `spotcheck` document. After grading, your validator waits for that document and checks both signatures. It then applies the measured figures and zeroes any miner with standing evidence, for the number of epochs the arena sets. A later report with no evidence for a miner does not erase a standing flag; only an upheld appeal lifts it. The server never bans anyone.

A miner may upload up to 3 times per epoch, so reports can name an older transcript. Your validator only uses a report's row for a miner when its `transcript_sha256` equals that miner's leaf in the `receipts` document; flags build up across reports for that transcript and older ones are ignored. A row whose `failure` begins `unchecked:`, or no row at all, means there is no spot check result: the miner scores zero in the specialist model and system pools for that epoch only, with no ban and no appeal, and your report shows the reason. A spot check that ran is a result, however few calls it reproduced. A failed memory or speed measurement counts against the SLM gates.

Your validator stops waiting once the coordinator has a spot check for at least one receipted transcript and has marked its report complete or covered every miner, or else at the spot check deadline, 150 blocks before the epoch ends. It falls back to a 5 hour limit only when the epoch window cannot be read. A read error is not a missing report: the validator reads again until the epoch ends. When the epoch has receipts and, at the deadline, no report carries a spot check for any receipted transcript (no document, a document that fails its signature, only reference rows, or only `unchecked:` rows), there is no coordinator report. The validator then keeps the last settled weights instead of setting new ones and its report says `coordinator_reported: false`. The server settles such an epoch void, with no new payout and no certificates, and the arena keeps paying the previous settlement. A miner's own timings are never used: without a measurement from the profile card, speed limits pass, speed earns no efficiency gain, and memory is computed from the model.

## Duplicates

Every validator runs this on its own, on CPU, and gets the same answer from the same files. Submissions go in commit order: the commit block from the roster, then the transaction's position in that block when blocks tie, then the hotkey if that position cannot be read. A sealed commitment counts from the sealed commit's block.

| Part | Duplicate of an earlier submission when | Pools zeroed |
|---|---|---|
| Model | Same weights hash, a change sketch similarity of at least 0.9, or at least 95% identical outputs over at least 20 common items | Specialist model and system |
| Router | Same weights hash, or a change sketch similarity of at least 0.9 | System |

The change sketch is a fixed 4096 number summary of the weights minus the arena's base model, so two fine tunes of one base are compared by what they changed. A model of another shape or a quantised model has no sketch and falls back to the hash and the outputs. A router only has a sketch when the arena definition names `references.router_base`. Harness and repair code is not compared, so miners may share it. A miner with a failure or a standing flag does not count as an earlier output for the behaviour comparison, because transcripts are public during the window.

On testnet, when the inputs bundle carries no reference harness, your validator and every miner build it from the miner template at the commit pinned in `microtensor_subnet/validator/items.py`, so both sides produce identical prompts.
