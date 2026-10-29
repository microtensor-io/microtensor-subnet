# Running a Microtensor validator

Netuid **576** on Bittensor testnet. A validator verifies and scores: every epoch it fetches each miner's transcript, replays the miner's artifact code against it, grades the answers once the labels are released, scores the five pools, signs a report and sets weights. Everything runs on CPU inside Docker. A validator serves no model and runs no miner model.

## Requirements

| | |
|---|---|
| Machine | one CPU box: 8 cores x86-64, 32 GB RAM, 200 GB NVMe |
| OS | Ubuntu 22.04 or 24.04 |
| Software | Docker 27 or newer |
| Network | a stable connection; every epoch the loop downloads each miner's transcript and the epoch bundles |
| Uptime | the loop must be running at each epoch's Freeze 2 and through the 6 hour grading window that follows |
| Access | a hotkey registered on netuid 576, a registered X25519 key, relay access granted by the team |

The only miner code a validator ever runs is the artifact's `main.py`, inside the CPU replayer image: read only, no network, a frozen clock, every capability dropped.

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

The labels bundle is sealed to a key only your validator holds. Register it, then ask the team to grant relay access to your hotkey (the relay pays for the reference checks your validator makes, within a per epoch allowance).

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

You choose nothing: every ten minutes the loop asks the server which arenas are live on your network, and the signed roster and mirror name the miners. When an arena reaches Freeze 2 it runs the seven steps below once; an epoch already reported is not run again, and the record is `/var/lib/microtensor/state/<arena>.epochs.json`.

## The seven steps of an epoch

1. **Load the epoch.** The arena definition, the signed `mirror`, `inputs` and `receipts` documents, the seed check against the chain's open block hash, the roster.
2. **Check receipts.** Fetch every miner's transcript from the mirror by its sha256; its leaf must sit under the `receipts` Merkle root. A miner with no upload is unscored.
3. **Replay.** For every miner and every item, run the frozen `main.py` from the mirror in the replayer image with no network and a frozen clock, answer each model call from the transcript, recompute every tool response from the bundle, verify every relay receipt, and require every request to match the recording exactly and in order. Count calls and tokens per condition; time the harness on your own clock. Any mismatch fails the miner for the epoch. Minutes, no model inference.
4. **Apply evidence.** Fetch the fraud evidence published for the epoch. A miner whose evidence has reached the arena's confirmation count scores zero and is banned for the arena's number of epochs.
5. **Grade.** After the labels bundle releases, open it with your key and grade every answer with the arena grader; the probe grader runs the arena's pinned small GGUF model with llama.cpp.
6. **Router pass.** Load each router from the mirror and score it on the kept drafts.
7. **Score, set weights, report.** The scoring formulas, gates and commit reveal. The report carries the receipt root and each miner's replay result; the server folds reports into the stake weighted median.

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

## Spot checks, a separate GPU duty

Spot checks are not part of validating and need nothing above. They are run by checkers holding a GPU of an approved hardware profile (`P1` RTX 4090, `P2` A100, `P3` H100, `K` T4 x2) with the NVIDIA Container Toolkit installed. A checker registers its profile, which is read from `nvidia-smi` or taken from `MT_PROFILE`:

```bash
docker run --rm -it --gpus all \
  -v /var/lib/microtensor:/var/lib/microtensor \
  -v ~/.bittensor/wallets:/wallets:ro \
  -e MT_WALLET_PATH=/wallets \
  --entrypoint mt ghcr.io/microtensor-io/validator:1 \
  epoch register-profile --wallet.name <coldkey> --wallet.hotkey <hotkey> --home /var/lib/microtensor
```

and starts the same loop with the GPU override, setting `MT_GPUS` in `.env`:

```bash
docker compose -f docker-compose.validator.yml -f docker-compose.gpu.yml up -d
```

Between replay and grading it then, for its share of the miners on its profile, pulls the weights from the mirror, starts the pinned engine exactly as the miner's runner did, re-runs at least 200 sampled model calls (100 uniform, 100 weighted), requires an exact token match, times them against the reference model on the same card, and publishes any mismatch as evidence. Validators apply that evidence in step 4. The owner keeps a checker on every profile its arenas allow.
