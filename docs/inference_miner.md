# Serving certified systems

Netuid **576** on Bittensor testnet. An inference operator holds the certified system of one or more arenas on its own GPUs and answers customer requests that the gateway at api.microtensor.cloud routes to it. Serving is a separate role from model mining and validating, with its own hotkey and its own settlement.

## What you run

Each held system is the four part artifact its miner submitted and validators certified: a small model served by vLLM, a router served as a classifier, and the miner's harness and repair code running in a gVisor sandbox with no network beyond four service URLs. Your host runs the same stack a validator runs, one system per GPU, and dials out to the gateway over a websocket. Nothing listens on a public port.

## Requirements

| | |
|---|---|
| OS | Ubuntu 22.04 or 24.04 |
| GPU | one 24 GB card per held system (RTX 4090, L4, A10G or better), CUDA 12 driver |
| CPU and RAM | 8 cores and 32 GB per held system |
| Disk | NVMe, 100 GB per held system, under `/var/lib/microtensor` |
| Software | Docker 27 or newer, NVIDIA Container Toolkit, gVisor (`runsc`) registered as a Docker runtime |
| Network | outbound 443 only; one system downloads once per epoch |
| Stake | 1 TAO collateral to the pool coldkey |

Host preparation (Docker, the NVIDIA toolkit, gVisor, `/var/lib/microtensor`) is the same as for a validator: follow steps 1 of [validator_setup.md](validator_setup.md) and confirm `docker run --rm --runtime=runsc hello-world` and the CUDA check both pass.

## 1. Register and post collateral

```bash
docker run --rm -it -v ~/.bittensor/wallets:/wallets:ro -e MT_WALLET_PATH=/wallets \
  --entrypoint mt ghcr.io/microtensor-io/validator:1 \
  operator register --label "my node" --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

The answer names the collateral amount and the coldkey to pay. Transfer it, then report the extrinsic:

```bash
docker run --rm -it -v ~/.bittensor/wallets:/wallets:ro -e MT_WALLET_PATH=/wallets \
  --entrypoint mt ghcr.io/microtensor-io/validator:1 \
  operator collateral --reference <extrinsic hash> --block <block> --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

## 2. See what the network wants

```bash
docker run --rm --entrypoint mt ghcr.io/microtensor-io/validator:1 operator plan --gpus 0,1 --gpu-gb 24
```

The plan reads the public catalogue (`GET /v1/public/inference/catalogue`) and ranks each arena's current system by demand: requests in the last hour divided by operators online, cold systems first, held systems kept unless something clearly wants serving more. One system per GPU; a system whose `vram_gb` exceeds `--gpu-gb` is skipped with the reason.

## 3. Run

```bash
cd microtensor-subnet/deploy
cat > .env <<EOF
MT_WALLET_NAME=<coldkey>
MT_WALLET_HOTKEY=<hotkey>
MT_GPUS=0
MT_GPU_GB=24
DOCKER_GID=$(getent group docker | cut -d: -f3)
EOF
docker compose -f docker-compose.operator.yml up -d
docker compose -f docker-compose.operator.yml logs -f
```

For one pinned system instead of the planner, run `operator run --serve mt/<arena>@<artifact sha prefix>` with the pin from the catalogue.

On start the host fetches the system's two repositories at the committed revisions, verifies every manifest hash, starts vLLM for the model, the router server and the sandbox, declares the system to the server (`POST /v1/inference/operators/models`) and dials the gateway. Expect ten to twenty minutes before the first answer on a fresh cache. Every `--review-seconds` (300) the catalogue is read again and the plan reconciled.

## 4. Admission

Validators probe operators through the gateway with withheld items the certified system answered correctly and grade your answers. A sequential test admits a clean operator after about sixty probes and rejects one that fails early; a rejection withdraws the operator until a new generation. Until validators probe, the team can admit an operator by hand for a demo. A new epoch winner resets eligibility for that model, so re-admission after each epoch is normal.

## How a request is answered

The gateway sends `{task, floor, max_tokens}` over the websocket. The host registers the task's schema as a SQLite database for the sandbox tools, calls the harness (`POST /harness/run`), asks the router for the probability that the draft is right, and answers `{answer, p, floor_met, verified, trace, usage}`. `floor_met` is whether the harness's own check passed and `p` cleared the customer's quality floor. No frontier call happens on an operator.

## Earnings

Serving pays from the serving pool by metered, billed usage per operator and model, gated on success rate, time to first token and time per token (`mt operator audit` recomputes the published settlement). The launch is meter only; the gates and the split are unchanged from the serving settlement.

## Troubleshooting

- `the docker socket is not reachable`: mount `/var/run/docker.sock` and set `DOCKER_GID`.
- `gVisor (runsc) is not registered`: `sudo runsc install && sudo systemctl restart docker`.
- `the model server did not become ready`: the card is too small for the model or another process holds it; `nvidia-smi` on the host.
- `this operator does not hold that system`: the gateway routed a model the plan dropped; it recovers at the next review.
