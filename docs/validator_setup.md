# Running a Microtensor validator

Netuid **576** on Bittensor testnet. One validator evaluates every submission of an arena each epoch on its own GPUs, reports the five pool scores to the server, and sets weights itself.

## Requirements

| | |
|---|---|
| OS | Ubuntu 22.04 or 24.04 |
| GPU | 3 × 24 GB (RTX 4090, L4, A10G or better), CUDA 12 driver. Arenas graded by a probe agent (context compression) place it automatically, see [Probe agent placement](#probe-agent-placement) |
| CPU | 16 cores |
| RAM | 64 GB |
| Disk | NVMe, 400 GB free. Bases and reference models stay cached between epochs; a miner's artifact and model are deleted as soon as its result is posted |
| Network | 1 Gbps. One epoch downloads every submission's model once |
| Software | Docker 27 or newer, NVIDIA Container Toolkit |
| Uptime | continuous. An epoch is 50,400 blocks (about a week); the evaluation window opens 7,200 blocks in and a full pass over 30 submissions takes about 20 hours on three GPUs |
| Access | a registered key and relay access granted by the team (see step 3) |

The sandbox is a locked down Docker container: no network except the gateway to four validator services, CPU, memory and process limits, a read only root file system, the artifact and the model mounted read only, every capability dropped and no privilege escalation. The validator itself runs as a container that starts sibling containers through the Docker socket: one vLLM server per submission on one GPU, one gateway, one sandbox. Missing sibling images (vLLM, sandbox, gateway) are pulled automatically, with a few retries.

## 1. Host preparation

```bash
# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker

sudo mkdir -p /var/lib/microtensor && sudo chown 10001:10001 /var/lib/microtensor
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

The CUDA check must succeed before anything else; the validator runs the same image to read the GPUs it was given. `/var/lib/microtensor` has to exist at that exact path on the host: the validator bind-mounts it at the same path inside so the sibling containers it starts see the same files.

## 2. Register the hotkey

```bash
btcli subnet register --netuid 576 --subtensor.network test --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

## 3. Keys and access

Hidden items are sealed to a key only your validator holds. Register it, then ask the team to grant relay access to your hotkey (the relay pays for frontier calls your evaluations make, within a per epoch allowance).

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

The key lives at `/var/lib/microtensor/keys/x25519.key` and the relay token beside it. Back the directory up; losing the key means re-registering and waiting for the next epoch.

## 4. Run

```bash
git clone https://github.com/microtensor-io/microtensor-subnet
cd microtensor-subnet/deploy
cat > .env <<EOF
MT_WALLET_NAME=<coldkey>
MT_WALLET_HOTKEY=<hotkey>
MT_GPUS=0,1,2
DOCKER_GID=$(getent group docker | cut -d: -f3)
EOF
docker compose -f docker-compose.validator.yml up -d
docker compose -f docker-compose.validator.yml logs -f
```

You do not choose arenas or miners: every ten minutes the loop asks the server which arenas are live on your network, and the signed roster names the miners. When an arena's evaluation window opens it downloads the epoch bundle, opens it with your key, sets up the baselines, evaluates the submissions on the signed roster one at a time at their committed revisions, posts each miner's signed result as it finishes, sets weights, and posts the signed epoch report. An epoch already reported is not run again; the record is `/var/lib/microtensor/state/<arena>.epochs.json`.

## What one epoch does

Setup, once per arena, before any miner is touched:

1. Loads the signed release, checks its seed against the chain's open block hash, opens the hidden items with your key, and loads the roster and the last epoch's best scores.
2. Sets up the baselines: the reference parts and every allowed base come from the cache and are downloaded only when missing, the probe agent starts where the hardware allows, and the reference system, the frontier redo and each base run on the hidden items. If the frontier redo fails, for example because the relay has no credit, the escalation and system pools have no reference this epoch and score nothing for anyone; the log says why.
3. Builds the miner queue from the chain: commitments and reveals, then timing, roster and fee for each, earliest commit first. Anything that fails is dropped with its reason.

Per miner, one at a time, or one per GPU when there are several:

4. Fetches the artifact and the model at the committed revision and checks arena, base, size, format, licence and the parameters read from the weights. A copy of an earlier miner's artifact or weights is refused; the earlier commit wins.
5. Evaluates in the sandbox: the model on the hidden items with speed and memory, the harness and escalation, and the full system at each quality floor with the miner's own router.
6. Scores provenance against the cached base, the margin over the reference and over the declared base, a lower bound score for each pool, and the envelope (at most 1% of items over).
7. Posts the signed miner result, keeps a small record (scores, drafts as text, digests) in the epoch work folder, and deletes the artifact and the model before the next miner starts. Only the router folder is kept, for the router pass. A post that fails is retried at the end of the epoch.

At close, once the queue is empty:

8. Runs each kept router on the other miners' drafts and the reference drafts, posts the miner's result again with the router pool, and deletes the router.
9. Ranks each pool and sets weights.
10. Posts the signed epoch report with the relay receipts, saves the probe sets for inference miners, and clears the epoch work folder except the report and the probe sets. Base weights stay cached for the next epoch.

A submission that fails a check, crashes, times out or reaches for the network is recorded with its reason, its files are deleted, and it scores nothing for the epoch.

## Dry run

```bash
docker compose -f docker-compose.validator.yml run --rm validator epoch run --arena sql_gen --dry-run --item-limit 20 --home /var/lib/microtensor
```

Evaluates twenty items per test, posts no miner results and no report, and sets no weights. Use it after installation and after every image update.

## Probe agent placement

Context compression arenas grade with a probe agent model. The validator reads the GPUs listed in `MT_GPUS` and places the probe agent itself:

| GPUs | Probe agent | Model under test |
|---|---|---|
| two or more | the last GPU alone, full model | the other GPUs, one miner per GPU |
| one card of 40 GB or more | the same card, full model at memory share 0.45 | memory share 0.45 |
| one smaller card, testnet only | the same card, a quantised build of the same model at memory share 0.4 | memory share 0.5 |

On any other network a single card under 40 GB is refused with `context compression needs a second GPU or a 40 GB card for the probe agent`. When the probe agent shares the card, miners are evaluated one at a time. `MT_SLM_MEMORY_SHARE` overrides the share of the model under test when set.

## Updating

```bash
docker compose pull && docker compose -f docker-compose.validator.yml up -d
```

Images are built from `main` and tagged `ghcr.io/microtensor-io/validator:1` and `ghcr.io/microtensor-io/sandbox:1`. The validator pulls the vLLM, sandbox and gateway images itself when they are missing.
