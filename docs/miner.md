# Running your own evaluation

Miners evaluate their own submissions and validators verify them. You commit a model and an artifact before Freeze 1, declare the hardware profile you will run on, and during the evaluation window you run one batch job on your own GPU. That job records every model, tool and relay call into a transcript, signs it with your hotkey and uploads it before Freeze 2. Validators replay the transcript on CPU and grade it after the labels release. A coordinator run by the subnet owner spot checks sampled model calls on a GPU, and validators read its signed reports from the server. The coordinator is the only thing in the evaluation that uses a GPU besides your own job; validators run on CPU. Nothing timed on your machine is used for scoring.

## Hardware profiles

A profile is one GPU architecture, one driver line and one image digest. Exact token matching only works within a profile, so declare the one you will really run on.

| Profile | GPUs | Precision | Batching |
|---|---|---|---|
| P1 | RTX 4090 only | bf16 | one request at a time |
| P2 | A100 80 GB (sm80) | bf16 | invariant |
| P3 | H100 (sm90); H200 after the pair test | bf16 | invariant |
| K | T4 x2 (sm75), small models only | fp16 | single |
| C | CPU, GGUF | gguf | single |

The arena lists the profiles it accepts in its definition. On P1 the runner reads the card name from `nvidia-smi` and refuses anything but an RTX 4090. Other cards of the same architecture can produce other tokens, and a token mismatch on the spot check can get an honest miner banned, so nothing overrides this check.

## What your machine needs

- One NVIDIA RTX 4090 with driver 580 or newer. No other card is accepted.
- Docker with the NVIDIA container toolkit, and your user able to run `docker compose`.
- 8 CPU cores, 32 GB of RAM and 100 GB of free disk under `/var/lib/microtensor`, for the images, your weights and the work folder.
- A steady connection to `api.microtensor.cloud`. The job waits out short outages, and you can resume it after a long one.

The job takes about 6 to 11 hours on an RTX 4090. The server publishes 920 items for `ctx_compress`, and each item runs one request at a time under every condition it belongs to.

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

The job also has to read your hotkey, and bittensor writes that file readable only by you. Copy only the hotkey and `coldkeypub.txt` to a folder owned by uid 10001. Your coldkey itself stays out of it, because the job signs with the hotkey alone:

```bash
export MT_WALLET_NAME=<coldkey> MT_WALLET_HOTKEY=<hotkey> MT_WALLET_DIR=/var/lib/microtensor-wallets
sudo mkdir -p "$MT_WALLET_DIR/$MT_WALLET_NAME/hotkeys"
sudo cp ~/.bittensor/wallets/$MT_WALLET_NAME/hotkeys/$MT_WALLET_HOTKEY "$MT_WALLET_DIR/$MT_WALLET_NAME/hotkeys/"
sudo cp ~/.bittensor/wallets/$MT_WALLET_NAME/coldkeypub.txt "$MT_WALLET_DIR/$MT_WALLET_NAME/"
sudo chown -R 10001:10001 "$MT_WALLET_DIR"
sudo find "$MT_WALLET_DIR" -type d -exec chmod 700 {} +
sudo find "$MT_WALLET_DIR" -type f -exec chmod 600 {} +
```

If the job stops with `wallet ... could not be unlocked`, this is the permission problem it is telling you about: uid 10001 can't read the files `MT_WALLET_DIR` points to.

Tell the job which group owns the Docker socket, pull the images and run it:

```bash
export DOCKER_GID=$(stat -c %g /var/run/docker.sock) MT_ARENA=ctx_compress
docker compose -f deploy/docker-compose.miner.yml pull
docker compose -f deploy/docker-compose.miner.yml up
```

or from a checkout with the validator extras installed:

```bash
mt miner run --arena ctx_compress --epoch <n> --home /var/lib/microtensor
```

The runner:

1. Reads the arena definition, the mirror and the inputs documents; refuses unless you are `mirrored` and your profile is allowed.
2. Pulls your own artifact and model from the mirror, verifies every digest, and runs the same pre-flight check as `mt check-model`.
3. Starts the pinned engine (`evaluation.images.engine`, by digest) with `--seed 0`, the profile's dtype, prefix caching off and logprobs on. On P1 it runs one request at a time, with no batch invariance. On P2 and P3 it sets `VLLM_BATCH_INVARIANT=1`.
4. Starts the recording gateway and your `main.py` in the pinned sandbox image with a frozen clock, no network except the gateway, and the item seed passed as `seed` in every request.
5. Runs every pool in a fixed order: SLM originals then twins, harness under your harness then `B0`, `B1`, `B2`..., escalation repairs from the published drafts, system items at each quality floor. Tool calls are served locally and deterministically; relay calls go through the owner relay with your miner relay token. Each item gets one relay ticket, shared by all its quality floors, with an equal share of your epoch allowance as its cap. A relay that answers 429 or 5xx, or doesn't answer, is retried with backoff for about 30 minutes, so a server deploy doesn't end the job. If the relay refuses a planned ticket the run stops with an error and uploads nothing, because a transcript with missing rows would lose those pools.
6. Writes the transcript, signs it, uploads it and prints the receipt. `--dry-run` skips the upload, writes `transcript.dry.json` instead and asks for no relay token or tickets, so it spends nothing; its repair and system items run with frontier calls refused. The transcript stores each request and router request with a sha256 in place of the full text, and no logprobs, so it fits the server's limits of 32 MiB compressed and 128 MiB of JSON.

The transcript keeps the item seed as published. The engine accepts signed 64 bit seeds, so the runner and the coordinator both send it the seed's low 63 bits.

The gateway forwards only `model`, `prompt`, `max_tokens`, `stop`, `temperature` and `stream` to the engine, with the temperature set to 0 and the seed set to the item's. Other sampling parameters your harness sends are dropped, because the transcript does not record them and the spot check could not reproduce them.

Progress is posted to the server as it runs. You can upload at most 3 times per epoch, and the server refuses a fourth. Sending the same transcript again doesn't use an upload. To replace an upload with a new run, delete `rows.jsonl` and `transcript.json` from the work folder and run the job again before Freeze 2. The new transcript replaces the old one, and the old one is ignored from then on: the coordinator drops its job for the old transcript, even one already running, and draws a new sample from the new one. Validators only read results for the transcript under the Freeze 2 receipts root. After Freeze 2 uploads are closed and a missing transcript leaves you unscored for the epoch.

## If something fails during the run

Each row is checked when it finishes. The runner redoes a row with a fresh recording when:

- the engine answered with an error, or with text but no token ids;
- `main.py` died, in which case it is started again;
- the row took longer than 600 seconds, in which case `main.py` is killed and started again;
- a model or relay call was still running when `main.py` answered, which is what a timed wait in your code looks like.

If the engine itself is gone, the runner starts it again first. A row is tried 3 times, and after that the job stops with an error and uploads nothing, because that row would fail replay and cost you the whole epoch. The error names the row and the reason.

## Stopping and resuming after a crash

The job appends each finished row to `rows.jsonl` in the work folder, `/var/lib/microtensor/work/<arena>/<n>/`, as soon as the row is done. To resume after a crash, a reboot or a stop, run the same command again. It reuses the finished rows when the arena, epoch, manifest, weights and inputs bundle all match, redoes only the row that was in flight, and takes the relay tickets it still needs again. Your transcript comes out the same as it would have without the crash. If any of those five changed, it starts over.

`docker compose stop` gives the job about 2 minutes to remove its containers. If a machine dies before that, the next start removes whatever the dead run left behind, so nothing keeps holding the GPU. If you run two jobs on one machine, give each its own `MT_SANDBOX_LABEL`, because the start of one job removes everything under its label.

## Sending a finished transcript

The upload retries with a fresh signature and growing pauses until uploads close whenever the server answers 429 or 5xx, times out, or can't read the chain. Any other refusal stops the job, and the transcript stays in the work folder. Send it without running anything again:

```bash
mt miner upload --arena ctx_compress --epoch <n> \
  --transcript /var/lib/microtensor/work/ctx_compress/<n>/transcript.json
```

With Docker, put the same arguments after the service name:

```bash
docker compose -f deploy/docker-compose.miner.yml run --rm miner miner upload \
  --arena ctx_compress --epoch <n> \
  --transcript /var/lib/microtensor/work/ctx_compress/<n>/transcript.json
```

## Determinism rules for artifact code

Validators replay your `main.py` on CPU and answer every call from your transcript. Requests must match exactly, in order. Code that breaks a rule fails replay and scores zero for the epoch.

- No clock reads beyond the frozen clock.
- Randomness only from the provided seed (`seed` in the request body).
- No threads, timers or concurrency that can reorder calls; issue calls in a fixed order per item.
- No sleeps and no timed waits. Replay answers every call at once, so a wait or a timeout that fires while recording doesn't fire during replay and the calls stop matching. Give model calls no timeout of your own; the runner ends a row at 600 seconds.
- No state kept between requests, because a resume or a redone row starts `main.py` fresh.
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
| Weight files | not enforced | with a `model.safetensors.index.json`, only the shards it names are read, because the engine loads only those; a weight file it doesn't name fails the SLM gate |
| Max input and output tokens | engine `--max-model-len`, gateway `max_tokens` | your `config.json`, then counted per call during replay |
| Model calls per item | gateway answers 429 past the arena's limit | counted from the transcript |
| Harness added wall time | not enforced | measured on the validator clock during replay |
| Harness and repair code size | `mt submit validate` | summed from the mirrored files; over the limit scores zero in the harness, escalation and system pools |
| Router parameters | `mt submit validate` | safetensors headers on the mirrored router; over the limit scores zero in the router pool |
| TTFT, TPOT, memory | not enforced | coordinator on the profile card against the reference on the same card |
| Relay spend | ticket cap from your epoch allowance | signed receipts summed |
| Tool faults | injected from the epoch seed | recomputed and compared |

Your own timings are never used. Without a measurement from the profile card, speed limits pass, speed earns no efficiency gain, and memory is computed from the model.

The escalation pool compares your repairs with the reference repairs on the same items. When no reference repair is published, the reference score is 0, the score of a full frontier redo, and the 5th percentile of your score over 200 family resamples must be above 0.

If a call differs from your transcript, the coordinator restarts its engine and reruns it. When both runs agree with each other and differ from yours, the call goes in its signed report as public evidence. Validators then zero your hotkey for the number of epochs the arena sets. Calls that faulted are not checked.

When the coordinator could not check you, for example because its GPU failed, its report says `unchecked` with the reason. That does not zero you. It means no spot check result this epoch, and the validators' reports show it.

You can appeal within one epoch. List your flagged calls, then appeal one, and the coordinator reruns it. A match lifts the flag.

```bash
mt miner appeal --arena <arena> --epoch <epoch>
mt miner appeal --arena <arena> --epoch <epoch> --item <item_id> --seq <seq>
```

Pass `--kind canary` to appeal a canary flag; the list shows each flag's kind.

## Duplicates

Every validator compares each submission's models with all earlier ones before scoring. The earliest commit block on chain wins. If two commits land in the same block, the earlier transaction in that block wins. A sealed commitment counts from the block of the sealed commit, not the reveal. Different hotkeys and coldkeys make no difference, because the check compares the files themselves.

| Part | Checked by | A duplicate when | Scores zero in |
|---|---|---|---|
| Model | Hash of the weights | Same hash as an earlier submission | SLM and system pools |
| Model | A fixed sketch of your weights minus the arena's base model | Similarity of at least 0.9 with an earlier sketch | SLM and system pools |
| Model | Your SLM outputs on the same items, token for token, against an earlier miner whose spot check sampled SLM calls and found no mismatch | At least 95% identical over at least 20 common items | SLM and system pools |
| Router | Hash of the weights, and the same sketch when the arena names a router base | Same hash, or similarity of at least 0.9 | Router and system pools |

Your other parts are still scored. Harness and repair code is never compared: two miners may use the same code, and only the model and the router have to be your own. Quantised models have no sketch, so only their hash and outputs are compared.
