# Running your own evaluation

Miners evaluate their own submissions and validators verify them. You commit a model and an artifact before Freeze 1, declare the hardware profile you will run on, and during the evaluation window you run one batch job on your own GPU. That job records every model, tool and relay call into a transcript, signs it with your hotkey and uploads it before Freeze 2. Validators replay the transcript on CPU and grade it after the labels release. A coordinator run by the subnet owner spot checks sampled model calls on a GPU, and validators read its signed reports from the server. The coordinator is the only thing in the evaluation that uses a GPU besides your own job; validators run on CPU. Nothing timed on your machine is used for scoring.

## Hardware profiles

A profile is one GPU architecture, one driver line and one image digest. Exact token matching only works within a profile, so declare the one you will really run on.

| Profile | GPUs | Precision | Batching |
|---|---|---|---|
| P1 | RTX 4090 (sm89); L4 and L40S after the pair test | bf16 | invariant |
| P2 | A100 80 GB (sm80) | bf16 | invariant |
| P3 | H100 (sm90); H200 after the pair test | bf16 | invariant |
| K | T4 x2 (sm75), small models only | fp16 | single |
| C | CPU, GGUF | gguf | single |

The arena lists the profiles it accepts in its definition. The runner detects the host profile from `nvidia-smi` and refuses to start when it differs from the one you committed; `MT_PROFILE=P1` overrides detection. A run with a remote engine skips this check, because the card is on the other machine; see [Using a remote engine](#using-a-remote-engine).

## Before you commit

Dry load the model the way the mirror and the runner will:

```bash
mt check-model --slm ./slm_repo --profile P1 --arena ctx_compress
```

It prints `ok` or one of `config: <reason>`, `tokenizer: <reason>`, `weights: <reason>`, `format: <reason>`, `size: <reason>`, plus the parameter count and the `weights_sha256` the manifest will carry. A model that fails here is excluded by the mirror at Freeze 1.

Write the manifest with the profile. The runner image digest is read from the arena definition:

```bash
mt submit manifest --artifact ./artifact_repo --arena ctx_compress \
  --slm ./slm_repo --slm-repo <user>/mt_slm_ctx_compress --slm-revision <sha> \
  --profile P1
mt submit validate --artifact ./artifact_repo --slm ./slm_repo
mt submit commit --arena ctx_compress --hf-user <user> --artifact-sha <sha> --profile P1
```

`--profile` writes manifest schema 2 (`profile`, `weights_sha256`, `image_digest`) and commits with the `mt4` (open) or `mt5` (sealed) tag. A manifest whose profile differs from the commitment is refused by the mirror.

## The evaluation window

A sealed commitment is revealed with `mt submit reveal --arena <arena>` at any time from the commit until Freeze 2. A reveal sent before Freeze 1 shows as unsealed at once and is carried onto the roster when it is built; keep both repositories private until Freeze 1, then make them public. `reveal_blocks` after Freeze 1 the server mirrors every revealed submission by content hash and retries a `missing` repository every ten minutes until Freeze 2. Check your entry at `GET /v2/arenas/<arena>/epochs/<n>/mirror` under `body.miners.<hotkey>`; only `mirrored` miners can run. Three blocks after the mirror pass the inputs bundle (items, seeds, databases, reference harnesses, escalation drafts) is published, with no labels. When the bundle carries no escalation drafts, the coordinator publishes them during the window. A run with the escalation part refuses to start until they are complete, so you do not spend hours on a transcript with no repair rows. On testnet a bundle may carry no reference harness; the runner then builds it from the miner template at the commit pinned in `microtensor_subnet/validator/items.py`, and validators build the same one, so both sides produce identical prompts. A commitment never revealed is `missed` at Freeze 2.

The job runs as uid 10001 and writes to `/var/lib/microtensor` on the host. Create that folder first, because Docker otherwise creates it owned by root and the job cannot write to it:

```bash
sudo mkdir -p /var/lib/microtensor && sudo chown 10001:10001 /var/lib/microtensor
```

Run the job:

```bash
MT_WALLET_NAME=<coldkey> MT_WALLET_HOTKEY=<hotkey> MT_ARENA=ctx_compress \
  docker compose -f deploy/docker-compose.miner.yml up
```

or from a checkout with the validator extras installed:

```bash
mt miner run --arena ctx_compress --epoch <n> --home /var/lib/microtensor
```

The runner:

1. Reads the arena definition, the mirror and the inputs documents; refuses unless you are `mirrored` and your profile is allowed.
2. Pulls your own artifact and model from the mirror, verifies every digest, and runs the same pre-flight check as `mt check-model`.
3. Starts the pinned engine (`evaluation.images.engine`, by digest) with `--seed 0`, the profile's dtype, `VLLM_BATCH_INVARIANT=1` on batch-invariant profiles, prefix caching off and logprobs on.
4. Starts the recording gateway and your `main.py` in the pinned sandbox image with a frozen clock, no network except the gateway, and the item seed passed as `seed` in every request.
5. Runs every pool in a fixed order: SLM originals then twins, harness under your harness then `B0`, `B1`, `B2`..., escalation repairs from the published drafts, system items at each quality floor. Tool calls are served locally and deterministically; relay calls go through the owner relay with your miner relay token. Each item gets one relay ticket, shared by all its quality floors, with an equal share of your epoch allowance as its cap. If the relay refuses a planned ticket the run stops with an error and uploads nothing, because a transcript with missing rows would lose those pools.
6. Writes the transcript, signs it, uploads it and prints the receipt. `--dry-run` skips the upload and asks for no relay token or tickets, so it spends nothing; its repair and system items run with frontier calls refused. The transcript stores each request and router request with a sha256 in place of the full text, and no logprobs, so it fits the server's limits of 32 MiB compressed and 128 MiB of JSON.

The transcript keeps the item seed as published. The engine accepts signed 64 bit seeds, so the runner and the coordinator both send it the seed's low 63 bits.

The gateway forwards only `model`, `prompt`, `max_tokens`, `stop`, `temperature` and `stream` to the engine, with the temperature set to 0 and the seed set to the item's. Other sampling parameters your harness sends are dropped, because the transcript does not record them and the spot check could not reproduce them.

Progress is posted to the server as it runs. You can upload at most 3 times per epoch, and the server refuses a fourth. To reupload, run the job again before Freeze 2. The new transcript replaces the old one, and the old one is ignored from then on: the coordinator drops its job for the old transcript, even one already running, and draws a new sample from the new one. Validators only read results for the transcript under the Freeze 2 receipts root. After Freeze 2 uploads are closed and a missing transcript leaves you unscored for the epoch.

## Using a remote engine

Without a Docker GPU machine you can run the job on any Docker host and send the model calls to a GPU somewhere else:

```bash
mt miner run --arena ctx_compress --epoch <n> --home /var/lib/microtensor \
  --engine-url https://<your-engine-host> --engine-key <token>
```

`MT_ENGINE_URL` and `MT_ENGINE_KEY` set the same two values. With them set the runner starts no engine container. It asks the remote host to fetch your model from the mirror by hash, starts the engine there, sends every model call to it, and cleans the remote host when the run ends. Your weights never touch the machine running the job. Your router runs on that machine's CPU, so give it enough memory. Docker is still required there, because your `main.py` and the gateway run in the sandbox.

The remote host must run the arena's pinned engine image with a card of your committed profile, and a small worker in front of it that takes a bearer token. The worker answers `GET /health`, `GET /engine`, `GET /tasks/<id>`, `POST /prepare`, `POST /engine/start`, `POST /calls` and `POST /cleanup`. Use a host of your own for one run at a time, because the cleanup removes everything the host fetched.

The transcript format does not change, and the spot check still decides whether your tokens are honest. A remote engine on another card or another image fails the exact token match like any other wrong setup.

## Determinism rules for artifact code

Validators replay your `main.py` on CPU and answer every call from your transcript. Requests must match exactly, in order. Code that breaks a rule fails replay and scores zero for the epoch.

- No clock reads beyond the frozen clock.
- Randomness only from the provided seed (`seed` in the request body).
- No threads, timers or concurrency that can reorder calls; issue calls in a fixed order per item.
- No file or network access outside the gateway.

Test before uploading:

```bash
mt miner replay --arena ctx_compress --epoch <n> --home /var/lib/microtensor \
  --transcript /var/lib/microtensor/work/ctx_compress/<n>/transcript.json
```

It starts the same gateway in replay mode and your artifact in the replayer image, posts every recorded request and reports the first mismatch.

## What is checked

| Budget | Enforced by the runner | Checked by validators |
|---|---|---|
| Parameter bounds and profile memory | pre-flight refuses | safetensors headers on the mirrored weights |
| Weight format | pre-flight refuses | read from the mirrored files against the arena's `allowed_formats`; other tensor types fail the SLM gate |
| Max input and output tokens | engine `--max-model-len`, gateway `max_tokens` | your `config.json`, then counted per call during replay |
| Model calls per item | gateway answers 429 past the arena's limit | counted from the transcript |
| Harness added wall time | not enforced | measured on the validator clock during replay |
| Harness and repair code size | `mt submit validate` | summed from the mirrored files; over the limit scores zero in the harness, escalation and system pools |
| Router parameters | `mt submit validate` | safetensors headers on the mirrored router; over the limit scores zero in the router pool |
| TTFT, TPOT, memory | not enforced | coordinator on the profile card against the reference on the same card |
| Relay spend | ticket cap from your epoch allowance | signed receipts summed |
| Tool faults | injected from the epoch seed | recomputed and compared |

Your own timings are never used. Without a measurement from the profile card, speed limits pass, speed earns no efficiency gain, and memory is computed from the model.

If a call differs from your transcript, the coordinator restarts its engine and reruns it. When both runs agree with each other and differ from yours, the call goes in its signed report as public evidence. Validators then zero your hotkey for the number of epochs the arena sets. Calls that faulted are not checked.

When the coordinator could not check you, for example because its GPU failed, its report says `unchecked` with the reason. That does not zero you. It means no spot check result this epoch, and the validators' reports show it.

You can appeal within one epoch. List your flagged calls, then appeal one, and the coordinator reruns it. A match lifts the flag.

```bash
mt miner appeal --arena <arena> --epoch <epoch>
mt miner appeal --arena <arena> --epoch <epoch> --item <item_id> --seq <seq>
```

Pass `--kind canary` to appeal a canary flag; the list shows each flag's kind.

## Duplicates

Every validator compares each submission with all earlier ones, part by part, before scoring. The earliest commit block on chain wins. If two commits land in the same block, the earlier transaction in that block wins. A sealed commitment counts from the block of the sealed commit, not the reveal. Different hotkeys and coldkeys make no difference, because the check compares the files themselves.

| Part | Checked by | A duplicate when | Scores zero in |
|---|---|---|---|
| Model | Hash of the weights | Same hash as an earlier submission | SLM and system pools |
| Model | A fixed sketch of your weights minus the arena's base model | Similarity of at least 0.9 with an earlier sketch | SLM and system pools |
| Model | Your SLM outputs on the same items, token for token | At least 95% identical over at least 20 common items | SLM and system pools |
| Harness code | Code with comments, whitespace and names removed | Same hash, or at least 90% of your fragments found in an earlier submission | Harness and system pools |
| Repair code | The same | The same | Escalation and system pools |
| Router | Hash of the weights, and the same sketch when the arena names a router base | Same hash, or similarity of at least 0.9 | Router and system pools |

Your other parts are still scored. Fragments that are in the pinned miner template do not count, so keeping the template unchanged never makes you a copy of another miner who kept it too. Quantised models have no sketch, so only their hash and outputs are compared.
