# Running your own evaluation

Arenas whose definition carries `evaluation.mode: miner` are evaluated by the miners themselves. You commit a model and an artifact before Freeze 1, declare the hardware profile you will run on, and during the evaluation window you run one batch job on your own GPU. That job records every model, tool and relay call into a transcript, signs it with your hotkey and uploads it before Freeze 2. Validators replay the transcript on CPU, spot check sampled model calls on the same profile, and grade after the labels release. Nothing timed on your machine is used for scoring.

Arenas without the block work exactly as before: validators fetch and run everything themselves and `mt miner` is not needed.

## Hardware profiles

A profile is one GPU architecture, one driver line and one image digest. You can only be spot checked by a validator on the same profile, so declare the one you will really run on.

| Profile | GPUs | Precision | Batching |
|---|---|---|---|
| P1 | RTX 4090 (sm89); L4 and L40S after the pair test | bf16 | invariant |
| P2 | A100 80 GB (sm80) | bf16 | invariant |
| P3 | H100 (sm90); H200 after the pair test | bf16 | invariant |
| K | Kaggle T4 x2 (sm75), small models only | fp16 | single |
| C | CPU, GGUF | gguf | single |

The arena lists the profiles it accepts under `evaluation.profiles`. The runner detects the host profile from `nvidia-smi` and refuses to start when it differs from the one you committed; `MT_PROFILE=P1` overrides detection.

## Before you commit

Dry load the model the way the mirror and the runner will:

```bash
mt check-model --slm ./slm_repo --profile P1 --arena ctx_compress
```

It prints `ok` or one of `config: <reason>`, `tokenizer: <reason>`, `weights: <reason>`, `size: <reason>`, plus the parameter count and the `weights_sha256` the manifest will carry. A model that fails here is excluded by the mirror at Freeze 1.

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

After Freeze 1 the server mirrors every revealed submission by content hash. Check your status at `GET /v2/arenas/<arena>/epochs/<n>/mirror`; only `mirrored` miners can run. When the window opens the inputs bundle (items, seeds, databases, reference harnesses, escalation drafts) is published, with no labels.

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
5. Runs every pool in a fixed order: SLM originals then twins, harness under your harness then `B0`, `B1`, `B2`..., escalation repairs from the published drafts, system items at each quality floor. Tool calls are served locally and deterministically; relay calls go through the owner relay with your miner relay token.
6. Writes the transcript, signs it, uploads it and prints the receipt. `--dry-run` skips the upload.

Progress is posted to the server as it runs. Upload again before Freeze 2 to replace an earlier transcript; after Freeze 2 uploads are closed and a missing transcript leaves you unscored for the epoch.

## Determinism rules for artifact code

Validators replay your `main.py` on CPU and answer every call from your transcript. Requests must match exactly, in order. Code that breaks a rule fails replay and scores zero for the epoch.

- No clock reads beyond the frozen clock.
- Randomness only from the provided seed (`seed` in the request body).
- No threads, timers or concurrency that can reorder calls; issue calls in a fixed order per item.
- No file or network access outside the gateway.

Test before uploading:

```bash
mt miner replay --arena ctx_compress --epoch <n> \
  --transcript /var/lib/microtensor/work/ctx_compress/<n>/transcript.json
```

It starts the same gateway in replay mode and your artifact in the replayer image, posts every recorded request and reports the first mismatch.

## What is checked

| Budget | Enforced by the runner | Checked by validators |
|---|---|---|
| Parameter bounds and profile memory | pre-flight refuses | safetensors headers on the mirrored weights |
| Max input and output tokens | engine `--max-model-len`, gateway `max_tokens` | counted per call during replay |
| Model calls per item | gateway answers 429 on the fourth | counted from the transcript |
| Harness added wall time | not enforced | measured on the validator clock during replay |
| TTFT, TPOT, memory | not enforced | spot checker on your profile against the reference on the same card |
| Relay spend | ticket cap from your epoch allowance | signed receipts summed |
| Tool faults | injected from the epoch seed | recomputed and compared |

A spot check mismatch produces public fraud evidence and, once confirmed, a hotkey ban for the number of epochs the arena sets.
