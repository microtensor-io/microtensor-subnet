# Running a Microtensor validator

Netuid **576** on Bittensor testnet. One validator evaluates every submission of an arena each epoch on its own GPUs, reports the five pool scores to the server, and sets weights itself.

## Requirements

| | |
|---|---|
| OS | Ubuntu 22.04 or 24.04 |
| GPU | 3 × 24 GB (RTX 4090, L4, A10G or better), CUDA 12 driver. Arenas graded by a probe agent (context compression) take the last GPU in `--gpus` for it, so give them one more |
| CPU | 16 cores |
| RAM | 64 GB |
| Disk | NVMe, 400 GB free. Models are cached per revision and pruned after each epoch |
| Network | 1 Gbps. One epoch downloads every submission's model once |
| Software | Docker 27 or newer, NVIDIA Container Toolkit, gVisor (`runsc`) registered as a Docker runtime |
| Uptime | continuous. An epoch is 50,400 blocks (about a week); the evaluation window opens 7,200 blocks in and a full pass over 30 submissions takes about 20 hours on three GPUs |
| Access | a registered key and relay access granted by the team (see step 3) |

Everything miner code does runs inside a gVisor sandbox with no network except four validator services. The validator itself runs as a container that starts sibling containers through the Docker socket: one vLLM server per submission on one GPU, one gateway, one sandbox.

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

# gVisor
curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | sudo tee /etc/apt/sources.list.d/gvisor.list
sudo apt-get update && sudo apt-get install -y runsc
sudo runsc install
sudo systemctl restart docker

sudo mkdir -p /var/lib/microtensor && sudo chown 10001:10001 /var/lib/microtensor
docker run --rm --runtime=runsc hello-world
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Both test commands must succeed before anything else. `/var/lib/microtensor` has to exist at that exact path on the host: the validator bind-mounts it at the same path inside so the sibling containers it starts see the same files.

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
MT_ARENA=sql_gen_s1
MT_GPUS=0,1,2
DOCKER_GID=$(getent group docker | cut -d: -f3)
EOF
docker compose -f docker-compose.validator.yml up -d
docker compose -f docker-compose.validator.yml logs -f
```

The loop polls the chain every ten minutes. When an arena's evaluation window opens it downloads the epoch bundle, opens it with your key, fetches every submission on the signed roster at its committed revision, runs the five tests, posts the signed report, and sets weights. An epoch already reported is not run again; the record is `/var/lib/microtensor/state/<arena>.epochs.json`.

## What one epoch does with a submission

1. Verifies the fee on chain, when the epoch's roster records one, and every file hash in `manifest.json`.
2. Serves the miner's model with vLLM on one GPU, isolated on an internal network.
3. Serves the miner's router from the validator process.
4. Starts the miner's `main.py` in the sandbox with four service URLs and nothing else.
5. Runs the SLM, harness, router, escalation and system tests on the epoch's hidden items.
6. Grades every answer outside the sandbox and signs the report.

A submission that fails to build, crashes, times out or reaches for the network scores zero in every pool for the epoch, and the report says why.

## Dry run

```bash
docker compose -f docker-compose.validator.yml run --rm validator epoch run --arena sql_gen_s1 --dry-run --item-limit 20 --home /var/lib/microtensor
```

Evaluates twenty items per test, posts no report and sets no weights. Use it after installation and after every image update.

## Updating

```bash
docker compose pull && docker compose -f docker-compose.validator.yml up -d
```

Images are built from `main` and tagged `ghcr.io/microtensor-io/validator:1` and `ghcr.io/microtensor-io/sandbox:1`. The validator pulls the sandbox image itself.
