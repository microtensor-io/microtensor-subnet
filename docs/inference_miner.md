# Serving certified systems

Netuid **576** on Bittensor testnet. An inference operator holds the certified system of one or more arenas on its own GPUs and answers customer requests that the gateway at api.microtensor.cloud routes to it. Serving is a separate role from model mining and validating, with its own hotkey and its own settlement.

## What you run

Each held system is the four part artifact its miner submitted and validators certified: a small model served by vLLM, a router served as a classifier, and the miner's harness and repair code running in a gVisor sandbox with no network beyond four service URLs. Your host runs the same stack a validator runs, one system per GPU, and dials out to the gateway over a websocket. Nothing listens on a public port.

## Requirements

| | |
|---|---|
| OS | Ubuntu 22.04 or 24.04 |
| GPU | one 24 GB card per held system (RTX 4090, L4, A10G or better), CUDA 12 driver; general models state their own class |
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

## General models

Beside certified systems, an operator can serve general open weight chat models by name (GLM, Qwen, DeepSeek, Kimi and similar releases) over the same socket, which now speaks operator protocol 4. Such a model appears in the catalogue with `kind: "general"`, a repository pinned at a revision, a sha256 manifest of every file it needs, its format, engine settings, context, maximum output, tool calling, family, serving class, memory and licence.

| Format | Engine | Image |
|---|---|---|
| awq, fp8, safetensors | SGLang | `--sglang-image` or `MT_SGLANG_IMAGE` |
| gguf | llama.cpp server | `--llama-image` or `MT_LLAMA_IMAGE` |

Serving classes are `gpu_24gb`, `gpu_80gb` and `gpu_8x80gb`. Your class follows `--gpu-gb`. The planner keeps one holding per GPU whatever its kind and skips, with the reason, a model whose class or `vram_gb` your card cannot meet or whose engine has no image (pass `none` as an image to turn that engine off).

The host downloads only the files the manifest names (`--hf-token` or `HF_TOKEN` for gated repositories), checks every sha256 and refuses the model on any mismatch, then starts the engine on one GPU with the weights mounted read only and offline. Each answer streams as chunks (content, reasoning, tool call argument fragments) and ends with the full message, usage, and the prompt and completion token ids. The host caps `max_tokens` at the model's maximum output and refuses a prompt that does not fit the context with `context_length_exceeded`.

Every general model records its licence and a link. The host holds none until you pass `--accept-licences` (`MT_ACCEPT_LICENCES=1` in the compose `.env`); without it the planner skips every general model, names its licence and link as the reason, and keeps your GPUs for certified systems. `operator plan` takes the same flag.

```bash
mt operator run --kind model --model glm-4.6-awq --gpus 0 --gpu-gb 80 --accept-licences
```

Other flags: `--kind system|model|any` (default any), `--model NAME` and `--family NAME` (repeatable), `--mem-fraction` (share of the card the engine keeps, 0.88).

Not yet live: engine image pins, memory fractions and parser flags are proven per model on a GPU box first, together with verification of general models by canonical prefill and their admission probes. Until that proof the catalogue lists no general model and the planner holds certified systems only.

## How general models are verified

Validators probe each general model through the gateway with a published pool of chat items: a plain question, a JSON reply, one and two step tool calls against a small weather and local time schema, and two long notes that fit 8K and 32K windows. Tool items go only to models that call tools and long items only to models whose context holds them. Probes run at temperature zero.

Every answer is first checked for format. The finish reason must be `stop`, `tool_calls` or `length`; a tool call must name a tool the request offered and carry arguments that parse as a JSON object; an answer that ends for tool calls must carry one. A broken answer is reported as `format`.

A validator that holds the model then verifies the answer by canonical prefill. It renders the prompt with the model's chat template and compares your prompt token ids with it, rebuilds the completion from the returned content, reasoning and tool calls in the family's raw format with its end of turn marker, and runs one forward pass over prompt and completion on its own copy of the pinned weights (SGLang `/generate` with input logprobs, or llama.cpp in process for GGUF). At every generated position it measures how far the returned token sits below the model's top choice. A substituted model, a cheaper quantisation or an edited answer leaves the top choice where the pinned weights would not, and a mean margin above the calibrated threshold is reported as `cheat`. When your prompt ids differ from the validator's template, the completion cannot be rebuilt or the prefill fails, the probe is `unproven`: it is recorded and never moves admission either way. `cheat` and `format` count as failed probes in admission and in the `cheat_share` gate of the serving settlement.

Prefill verification shows that the text you returned is what the pinned weights produce greedily for that prompt. It does not cover sampled answers, latency (measured from metering) or models no validator class can hold. A model whose `verifiable_on` is empty (the largest releases) is probed by outcome and format only: an answer from the listed digest that passes the format checks counts as a pass, and the catalogue says so per model.

Calibration and the per model verifiability table wait for a GPU box. A threshold is measured per model and digest from honest answers and from a substitute model (`mt epoch probe --general MODEL --calibrate --substitute OTHER`, with `--verify-backend sglang|llama`, `--engine-url` and `--gpu`) and stored under the validator's `calibration/<model>/<digest>.json`; a validator without one refuses to probe that model, and a threshold that never separated a substitute leaves every verdict `unproven`. No general model is listed before those numbers exist.

## Earnings

Serving pays from the serving pool by metered, billed usage per operator and model, gated on success rate, time to first token and time per token (`mt operator audit` recomputes the published settlement). The launch is meter only; the gates and the split are unchanged from the serving settlement.

## Troubleshooting

- `the docker socket is not reachable`: mount `/var/run/docker.sock` and set `DOCKER_GID`.
- `gVisor (runsc) is not registered`: `sudo runsc install && sudo systemctl restart docker`.
- `the model server did not become ready`: the card is too small for the model or another process holds it; `nvidia-smi` on the host.
- `this operator does not hold that system`: the gateway routed a model the plan dropped; it recovers at the next review.
