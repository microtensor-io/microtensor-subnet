# Mining on Microtensor

You compress a frontier model into a specialist that fits a hardware class, host
the artifact somewhere validators can fetch it, and commit a 128-byte pointer on
chain each round. That is the whole job.

**You do not serve inference.** There is no axon, no request handler, no GPU that
has to stay warm, no uptime requirement. Validators fetch your artifact and run
it on their own certified hardware. A miner box that is offline between rounds
loses nothing.

If you are here to evaluate other people's models, read
[validator_setup.md](validator_setup.md) instead.

---

## 1 · What you are actually competing on

Two numbers decide everything, in this order:

1. **Does it fit?** Size, peak resident memory at your declared maximum input,
   and p95 time-to-first-token must all sit under your class ceiling. This is
   binary. Over on any axis and your model does not exist for the round.
2. **How accurate is it?** Only among models that fit.

A more accurate model that misses the memory ceiling scores zero against a worse
model that fits. Design for the envelope first.

| track | class | size | peak RSS | p95 TTFT | emission |
|---|---|---|---|---|---|
| `code` | `laptop` | 1.5 GiB | 3 GiB | 180 ms | 60 % |
| `code` | `edge-gpu` | 2.5 GiB | 4 GiB | 120 ms | 40 % |

```bash
mt inspect tracks
```

These are the two live competitions at launch. `code` carries the whole
emission, split 60/40 between `laptop` and `edge-gpu`. Each competition pays
its top 8 on a geometric curve, and rank 8 still earns about a third of
rank 1, so the tail is worth competing for. Further tracks and classes are
registered as disabled stubs and open by governance.

---

## 2 · Declare honestly, because the mechanism makes it your best move

You declare an envelope. The validator measures one. Both are checked:

```
measured > class ceiling      → inadmissible
measured > your declaration   → inadmissible
```

**Over-declaring does not help you.** Your declaration is published in the
Verified Model Certificate, and a deployer choosing between two admissible models
picks the one with the tighter guarantee. **Under-declaring is fatal.** A tolerance is applied to your declaration
only, never to the class ceiling: 2 % on size and memory, and on latency the
larger of 10 % or 15 ms, because single-digit-millisecond slack would punish
thermal jitter rather than lies.

The truthful declaration is the one that is barely above what you actually
measured. `mt miner selfcheck` computes exactly that, with a 10 % margin.

---

## 3 · Your rig

Nothing here is enforced or even visible to the subnet. You upload an artifact
and commit a pointer, so your hardware is unverifiable by construction. This is
what the work actually takes.

**The GPU is for building the model, not for running it.** Compression and
fine-tuning are the real cost, and they happen entirely off-subnet.

| | |
|---|---|
| **Recommended** | RTX 4090 24 GB, or RTX 5090 32 GB |
| **Floor** | RTX 3090 24 GB |
| **RAM** | 32 GB |
| **Disk** | 500 GB SSD |
| **Vendor** | **NVIDIA only** |

NVIDIA is not a preference. `bitsandbytes` and `unsloth` require CUDA, so AMD,
Apple Silicon and Intel GPUs will not run the recommended stack at all.

Use **Unsloth**. It cuts VRAM by 60 to 70 % and runs 2 to 5 times faster, which
is the difference between a 24 GB card being enough and not being enough.

Two smaller numbers, which people conflate with the above:

- **Running the microtensor CLI**: 2 cores, 4 GB RAM, no GPU. That is all
  `init`, `package`, `upload`, `publish` and `run` ever need.
- **Running `mt miner selfcheck`**: RAM above the ceiling of the class you
  target, so roughly 20 GB for `server-cpu`, 6 GB for `edge-gpu`, 5 GB for
  `laptop`, 3 GB for `embedded`. Still no GPU, since the engine is CPU-only.

Full breakdown in [min_compute.yml](../min_compute.yml).

---

## 4 · Install

Miners need almost nothing: no runtime and no profiler dependencies for the base
install:

```bash
git clone https://github.com/microtensor-io/microtensor-subnet
cd microtensor-subnet
python -m venv .venv && source .venv/bin/activate
pip install .
```

To run `selfcheck` locally, which is strongly recommended, add the runtime too:

```bash
pip install ".[validator]"
```

### Register

Microtensor is **netuid 92** on finney, and that is the built-in default.

```bash
btcli subnet register --netuid 92 --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

```bash
export MT_NETWORK=finney
export MT_WALLET_NAME=<coldkey>
export MT_WALLET_HOTKEY=<hotkey>
```

---

## 5 · Build the artifact

A directory containing everything needed to load and run the model:

```
my-model/
├── model.onnx        # the entrypoint named in your manifest
└── tokenizer.json
```

**Constraints that will reject you at submission:**

- Start from a base model on the pinned allowlist in
  [constants.py](../microtensor/core/constants.py), and declare it in your
  manifest as `<repo>@<revision-sha>`. The candidate list is Qwen3 (0.6B,
  1.7B, 4B) and Llama 3.2 (1B, 3B); the exact revisions are published at
  corpus freeze, and until the allowlist is frozen the field is unchecked.
- ONNX opset 13 to 21, standard domains only (`""`, `ai.onnx`, `ai.onnx.ml`). Custom
  operators are rejected, because a validator will not run code it cannot audit.
- Greedy decoding. The track fixes it; temperature is not a knob you control.
- Size is *total*: weights, tokenizer, config, everything in the directory.
- No `manifest.json` of your own. Packaging writes it.

### Training data

Each corpus release ships a public train split (`code.train.jsonl`: prompts
and public examples only) and one vetted reference completion per train task
(`code.reference.jsonl`). Fine-tune on those prompt/completion pairs
directly; nobody needs to run a large model over the corpus themselves. The
hidden tests and the rotating draw never leave the validator bundle.

---

## 6 · Self-check before you declare

```bash
mt miner selfcheck \
  --artifact ./my-model \
  --track code \
  --hardware-class laptop \
  --source hf:youracct/mt-code-3b@v1 \
  --profile-seconds 60
```

This runs the validator's own profiler against your artifact: cold start, then a
sustained stream at your declared maximum input, sampling RSS throughout and
keeping the maximum. Output:

```
class            laptop  (developer workstation)
size             1.31 GiB   ceiling 1.50 GiB
peak rss         2.48 GiB   ceiling 3.00 GiB
ttft p50 / p95   96 / 142 ms   ceiling 180 ms
cold start       1830 ms
throughput       48.6 tok/s
device profile   dev:9f2c1a0b4e7d8c33

declare this envelope:
  size_bytes      1546188226
  peak_rss_bytes  2932735283
  p95_latency_ms  157
```

If it prints `INADMISSIBLE`, fix the model. Do not declare around it.

> Your numbers will differ from the validator's; it measures on certified
> reference hardware. Leave headroom on any axis you are close to.

---

## 7 · Ship it

Everything above is one-time. Mining is four commands total:

```bash
btcli subnet register --netuid 92 --wallet.name <coldkey> --wallet.hotkey <hotkey>

mt miner init --artifact ./my-model --track code --hardware-class laptop \
              --source hf:youracct/mt-code-3b@v1

mt miner ship

pm2 start "mt miner run" --name microtensor-miner --kill-timeout 3000
```

`mt miner init` writes `~/.microtensor/miner.json` once. **After that every
command runs bare**, with no repeated flags. Any flag you do pass overrides the
saved value for that invocation only.

`mt miner ship` runs the whole pipeline: self-check if you haven't, package
(digest every file, sign the manifest with your hotkey), upload to your source,
and commit the pointer on chain. Uploading is real for `hf`, `s3` and `r2`; for
a plain `https` host publish the files with your own tooling and use
`mt miner publish` without `--upload`.

The commit is about 84 bytes:

```
mt1|41|code|laptop|3f9a…c21b|hf:youracct/mt-code-3b@v1
```

`mt miner run` then re-commits before every round closes, so you stay in the
competition without touching anything. It serves no traffic and holds no
inference. It is a scheduler, and it can go down between rounds without costing
you a thing.

### The individual steps, if you prefer them

```bash
mt miner selfcheck      # measure, propose a declaration
mt miner package        # digest + sign + write manifest.json
mt miner upload         # push files to your source
mt miner publish        # commit the pointer
mt miner status         # what you would publish right now
```

The source layout must match `<source>/<path>`:

```
hf:youracct/mt-code-3b@v1  →  model.onnx, tokenizer.json, manifest.json
```

Supported schemes: `hf`, `https`, `s3`, `r2`, `ipfs`.

---

## 8 · Timing

```
[start ─────────────────── close) [close ──────── deadline]
 publish inside this window        evaluation, too late to submit
```

Submissions close 600 blocks (≈ 2 h) before the round ends, and the participant
set freezes there by artifact digest. **A commitment for round *N* must land
before round *N*'s close block.** Anything later competes in round *N+1*.

The round seed is drawn from the hash of that very close block, so no submitter
can see the task set. Do not bother trying to time it.

### The reveal deadline

Your artifact must be **fetchable by validators at the close block**, because
that is when they materialise the participant set. If your repo is private
while you train, flip it public (or grant the validator token) BEFORE the
close, not after. The chain also rate-limits commitments to roughly one per
20 minutes, so the last safe commit is well before the close. Against
`SUBMISSION_CLOSES_BEFORE_BLOCKS = 600` (the close sits about 2 hours before
round end), a sane timeline for a 24-hour round looks like:

- T-4h: artifact uploaded, `mt miner selfcheck` clean
- T-3h: repo public / token granted, `mt miner ship`
- T-2h: submissions close; the seed is drawn; too late for this round

Prefer `hf:<org>/<repo>@<commit-sha>` over `https:` sources. The digest
always binds the bytes, but a pinned commit also survives the host moving
the branch under you.

---

## 9 · What gets you zeroed

| | |
|---|---|
| unsigned manifest, or signed by a different hotkey | rejected at discovery |
| manifest does not hash to the committed digest | rejected at discovery |
| commitment names the wrong round or a closed competition | rejected at discovery |
| a file's digest does not match the manifest | scores 0 |
| unlisted files in the fetched tree | scores 0 |
| custom operators or an unsupported format | rejected at submission |
| over any class ceiling, or over your own declaration | inadmissible, scores 0 |
| a task times out or the process dies | that task scores 0 |
| flagged as a copy of an earlier artifact | the later submission scores 0 |

Note the asymmetry on the last one: publishing early protects you, since the copy
detector zeroes the *later* submission.

`mt inspect hotkey <your-ss58>` on any validator shows exactly which of these hit
you and why.

---

## 10 · Where the wins are

- **Quantise past where it looks safe.** The gate is on the envelope, not on
  parameter count. int8 or int4 that holds accuracy beats fp16 that misses RSS.
- **Peak RSS is measured at your declared maximum input, under sustained load.**
  not at load, not at steady state. For autoregressive models the KV cache grows
  linearly in context. Declare a maximum you can actually hold, then hold it.
- **Cold start is measured on an unwarmed cache.** A single warm measurement is
  precisely what a miner would construct if permitted to, so it is not admissible.
- **Rank 8 pays.** The curve is geometric with decay 0.85, so do not concede a
  competition because you cannot take rank 1.
- **Incumbency decays.** Resubmitting the same digest round after round bleeds
  share to active positions. Ship improvements.
- **Hysteresis protects holders at every rank.** Beating an incumbent takes a
  margin of 0.005, not a tie. Copying the leader to land just behind them does
  not work.
