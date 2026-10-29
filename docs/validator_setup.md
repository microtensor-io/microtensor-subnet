# Running a Microtensor validator

Netuid **576** on Bittensor testnet. Miners run their own evaluation on their own GPU and upload a transcript. A validator verifies every transcript: it replays the miner's code on CPU, re-runs a chain seeded sample of model calls on a GPU of the same hardware profile and requires the same tokens, grades after the labels release, reports the five pool scores to the server, and sets weights itself.

## Requirements

| | |
|---|---|
| OS | Ubuntu 22.04 or 24.04 |
| CPU box | 8 cores, 32 GB RAM, 200 GB NVMe; runs replay, grading, the router pass and the probe grader |
| GPU | one card of an approved hardware profile (`P1` RTX 4090, `P2` A100, `P3` H100, `K` T4 x2) for about 20 to 30 minutes per epoch; a rented machine is fine. No GPU at all is allowed: a CPU only validator replays, grades and confirms evidence but runs no spot checks |
| Network | a real connection; each sampled miner's weights are about 15 GB and the download dominates the spot check |
| Software | Docker 27 or newer; the NVIDIA Container Toolkit on a GPU box |
| Uptime | the loop runs after each epoch's Freeze 2; an epoch is about a week on testnet |
| Access | a registered key, relay access granted by the team, and a registered profile (see step 3) |

The only miner code a validator runs is the artifact's `main.py`, inside the CPU replayer image: read only, no network, a frozen clock, every capability dropped. No validator serves a miner's model.

## 1. Host preparation

```bash
# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit, GPU box only
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

sudo mkdir -p /var/lib/microtensor && sudo chown 10001:10001 /var/lib/microtensor
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi   # GPU box only
```

On a GPU box the CUDA check must succeed before anything else; the validator reads its profile from the same `nvidia-smi`. `/var/lib/microtensor` has to exist at that exact path on the host: the validator bind-mounts it at the same path inside so the sibling containers it starts see the same files.

## 2. Register the hotkey

```bash
btcli subnet register --netuid 576 --subtensor.network test --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

## 3. Keys, access and profile

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

Register the profile your hardware holds, so the spot check sample counts you and the miners on your profile are shared out among the validators that can check them:

```bash
docker run --rm -it --gpus all \
  -v /var/lib/microtensor:/var/lib/microtensor \
  -v ~/.bittensor/wallets:/wallets:ro \
  -e MT_WALLET_PATH=/wallets \
  --entrypoint mt ghcr.io/microtensor-io/validator:1 \
  epoch register-profile --wallet.name <coldkey> --wallet.hotkey <hotkey> --home /var/lib/microtensor
```

It detects the profile from `nvidia-smi` (compute capability 8.9 is `P1`, 8.0 is `P2`, 9.0 is `P3`, 7.5 is `K`, no GPU is `C`), or takes `MT_PROFILE`, and posts the signed list to the server. Leave out `--gpus all` on a CPU box. Register again when the hardware changes.

The key lives at `/var/lib/microtensor/keys/x25519.key` and the relay token beside it. Back the directory up; losing the key means re-registering and waiting for the next epoch's labels release.

## 4. Run

```bash
git clone https://github.com/microtensor-io/microtensor-subnet
cd microtensor-subnet/deploy
cp .env.example .env   # wallet names, MT_GPUS, DOCKER_GID
docker compose -f docker-compose.validator.yml -f docker-compose.gpu.yml up -d
docker compose -f docker-compose.validator.yml -f docker-compose.gpu.yml logs -f
```

A CPU only box leaves out `docker-compose.gpu.yml`; that override is what hands the validator its GPUs. You do not choose arenas or miners: every ten minutes the loop asks the server which arenas are live on your network, and the signed roster and mirror name the miners. When an arena reaches Freeze 2 it runs the eight steps below once; an epoch already reported is not run again, and the record is `/var/lib/microtensor/state/<arena>.epochs.json`.

## The eight steps of an epoch

Only step 4 needs the GPU.

1. **Load the epoch.** The arena definition, the signed `mirror`, `inputs` and `receipts` documents, the seed check against the chain's open block hash, the roster.
2. **Check receipts.** Fetch every miner's transcript from the mirror by its sha256; its leaf must sit under the `receipts` Merkle root. A miner with no upload is unscored.
3. **Replay.** For every miner and every item, run the frozen `main.py` from the mirror in the CPU replayer image with no network and a frozen clock, answer each model call from the transcript, recompute every tool response from the bundle, verify every relay receipt, and require every request to match the recording exactly and in order. Count calls and tokens per condition; time the harness on your own clock. Any mismatch fails the miner for the epoch. Minutes, no model inference.
4. **Spot check.** Draw your share of the miners on your profile from the close block hash and your hotkey (about two validators per miner); for each, pull the weights from the mirror, start the pinned engine exactly as the miner's runner did, re-run at least 200 sampled model calls (100 uniform, 100 weighted) and require an exact token match, and record first token time, per token time and peak VRAM; time the reference model on the same card and re-check a few owner signed reference drafts. About 20 to 30 minutes, dominated by the downloads.
5. **Act on evidence.** Fetch the evidence other validators published for the epoch, re-run each cited call on your profile, and post a confirmation. Zero any miner whose evidence reaches the arena's confirmation count.
6. **Grade.** After the labels bundle releases, open it with your key and grade every answer with the arena grader on CPU; the probe grader runs the arena's pinned small GGUF model with llama.cpp.
7. **Router pass.** Load each router from the mirror and score it on the kept drafts.
8. **Score, set weights, report.** The scoring formulas, gates and commit reveal. The report carries the receipt root, your profile, and each miner's replay and spot check results. A miner whose replay failed scores zero; a miner with confirmed evidence scores zero and is banned for the arena's number of epochs.

A CPU only validator runs steps 1 to 3 and 5 to 8.

## Dry run

```bash
docker compose -f docker-compose.validator.yml -f docker-compose.gpu.yml run --rm validator epoch run --arena ctx_compress --dry-run --item-limit 20 --home /var/lib/microtensor
```

Replays and grades twenty items per pool, posts nothing and sets no weights. Use it after installation and after every image update.

## Updating

```bash
docker compose pull && docker compose -f docker-compose.validator.yml -f docker-compose.gpu.yml up -d
```

The validator image is built from `main` and tagged `ghcr.io/microtensor-io/validator:1`. The engine, replayer, sandbox and llama.cpp images are pulled by the validator itself, by the digests pinned in each arena's definition, never by tag alone.
