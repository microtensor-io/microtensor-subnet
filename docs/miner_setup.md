# Mining on Microtensor

You compress a frontier model into a specialist that fits a hardware class, host
the artifact somewhere validators can fetch it, and commit a 128-byte pointer on
chain each round. That is the whole job.

**Your machine trains; the network runs.** Validators fetch your artifact and
execute it on their own certified hardware, so your box is busy exactly while
you are training and improving the model. Once per round you come online inside
the submission window, send one extrinsic, and go back to work.

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
| `code` | `mt-3g` | 1.5 GiB | 3 GiB | 180 ms | 100 % |

```bash
mt inspect tracks
```

One live competition at launch, so every entrant is in the same contest rather
than split across thin ones. **A class is a memory envelope, not a device.**
`mt-3g` means 3 GiB peak resident memory at your declared maximum input, 1.5 GiB
on disk, and 180 ms p95 first output; it says nothing about what hardware you
train on. The competition pays its top 8 on a geometric curve, and rank 8 still
earns about a third of rank 1, so the tail is worth competing for.

`mt-4g`, `mt-16g` and `mt-1g` are registered and open by governance once real
submissions exist, in the same shape as the disabled track stubs.

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

Your rig is yours alone: you upload an artifact and commit a pointer, so the
subnet sees the model and nothing about the machine that made it. The figures
below are what the work actually takes.

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
  target, so roughly 20 GB for `mt-16g`, 6 GB for `mt-4g`, 5 GB for
  `mt-3g`, 3 GB for `mt-1g`. Still no GPU, since the engine is CPU-only.

Full breakdown in [min_compute.yml](../min_compute.yml).

---

## 4 · Install

The base install is the CLI on its own, which is all that packaging and
committing need:

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
- GGUF v2 to v3, or ONNX opset 13 to 21 with standard domains only (`""`,
  `ai.onnx`, `ai.onnx.ml`). `safetensors` has no engine and is not accepted;
  run `mt inspect engines` against a validator build to see what it will run. Custom
  operators are rejected, because a validator will not run code it cannot audit.
- Greedy decoding. The track fixes it; temperature is not a knob you control.
- Size is *total*: weights, tokenizer, config, everything in the directory.
- Leave `manifest.json` to `mt miner package`, which writes it for you.

### Training data

Each corpus release publishes a train split — prompts and public examples
only — from the read API:

```bash
curl https://api.microtensor.cloud/v1/corpora/<corpus-version>/public
```

The response carries the manifest, the per-partition counts, and the `train`
tasks themselves. The `fixed` and `rotating` partitions and every hidden test
stay on the control plane and are served only to validators, so what you can
read is exactly what you may train on.

The same response carries one vetted reference completion per train task
under `reference`, with `reference_model` naming what produced them, so you
can fine-tune on prompt/completion pairs without running a large model over
the corpus yourself:

```json
{"ref": "code-000341", "model": "<repo>@<sha>", "completion": "def parse_ledger(text): ..."}
```

`model` names the pinned model that produced the completion, so you know what
you are distilling from.

---

## 6 · Publish a training run

**This is required for admission.** A submission without a resolvable run is
rejected at discovery, in the same class as an unsigned manifest, and never
enters the round.

Log to the subnet's project, named for your hotkey:

```python
import time, wandb

wandb.init(
    entity="microtensor",
    project="training-runs",
    name="<your-hotkey-ss58>",
    config={
        "mt_track": "code",
        "mt_class": "mt-3g",
        "mt_base_model": "<repo>@<sha>",
        "mt_corpus_version": "<corpus digest>",
    },
)

# during training, whatever your run actually produced
wandb.log({"step": n, "loss": ..., "eval_pass_rate": ...})
```

The binding field is the artifact digest, which does not exist until you
package. So the order is **package, then log the digest, then ship**.
`mt miner package` prints the exact lines:

```python
wandb.run.summary["mt_artifact_digest"] = "sha256:3f9a1c04…c21b"
wandb.run.summary["mt_finished_at"] = 8894210   # the current block height
wandb.finish()
```

`mt_finished_at` is a **chain block height**, not a wall-clock time.
Validators check that your run finished at or before the block the round's
commitment window closes on, and block heights compare exactly where clock
conversions drift. `mt miner package` prints these lines with the live
height already filled in; a wall-clock timestamp here reads as a block far
in the future and is rejected.

Check it before committing anything:

```bash
mt miner provenance
```

```
run          microtensor/training-runs/5F3sa2TJ…
status       resolvable
digest       matches submitted artifact
base model   Qwen/Qwen3-1.7B@a1b2c3d  (on allowlist)
finished     block 8,894,210
verdict      ADMISSIBLE
```

`mt miner ship` and `mt miner publish` refuse to commit when this fails, so you
find out locally rather than being dropped silently at discovery.

A run is a record, not a proof. What it establishes is that you published a
training history bound to this artifact before you submitted it. What proves
the model is the network running it on its own hardware, plus the divergence
and uplift the validator measures from your weights directly.

---

## 7 · Self-check before you declare

```bash
mt miner selfcheck \
  --artifact ./my-model \
  --track code \
  --hardware-class mt-3g \
  --source hf:youracct/mt-code-3b@a1b2c3d \
  --profile-seconds 60
```

This runs the validator's own profiler against your artifact: cold start, then a
sustained stream at your declared maximum input, sampling RSS throughout and
keeping the maximum. Output:

```
class            mt-3g  (developer workstation)
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

## 8 · Ship it

Submitting grants the network the right to retain, archive and redistribute
any artifact it certifies, together with its manifest and measured record.
Certificates resolve against the network's archived copy, so deleting your own
hosting after settlement does not break them.

Everything above is one-time. Mining is five commands total:

```bash
btcli subnet register --netuid 92 --wallet.name <coldkey> --wallet.hotkey <hotkey>

mt miner init --artifact ./my-model --track code --hardware-class mt-3g \
              --source hf:youracct/mt-code-3b@a1b2c3d

mt miner package        # prints the artifact digest to log to your training run
mt miner provenance     # confirm the run resolves and binds to that digest
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
mt1|41|code|mt-3g|3f9a…c21b|hf:youracct/mt-code-3b@a1b2c3d
```

`mt miner run` then re-commits before every round closes, so you stay in the
competition without touching anything. It serves no traffic and holds no
inference. It is a scheduler, and it can go down between rounds without costing
you a thing.

### Sealed submissions

While a round is open, an unencrypted artifact on a public host can be copied
and committed by someone else before you. To close that window, submit sealed:

```bash
mt miner ship --sealed
```

This encrypts your artifact under a fresh key, publishes only the ciphertext,
and commits the same pointer marked sealed. Nothing readable is public during
the window. At the close block the key is revealed on chain, validators
decrypt, verify the plaintext against the digest you committed, and measure.

`mt miner run` posts the reveal for you the moment the round closes. The key
is held at `~/.microtensor/reveal/` on the machine that packaged the artifact;
keep that machine running through the close block, or reveal by hand from it:

```bash
mt miner reveal
```

**A sealed submission that is never revealed is excluded, with the reason
`sealed submission was never revealed`.** There is no penalty beyond missing
that round, and nothing to appeal: the key either arrived on chain in the
reveal window or it did not. If you ship sealed, keep the packaging machine up,
or reveal manually before the window closes.

The reveal is a second commitment that replaces your submission in your hotkey's
single on-chain slot. Everything a validator needed from the original was read
while the round was open, so the slot only has to carry the key afterward.

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

## 9 · Timing

```
[open ───────────────────── close) [close ──────── deadline]
 publish inside this window         evaluation, too late to submit
```

The operator opens each round's submission window, and your client reads the
open round and its close block from the network automatically. You set nothing:
`mt miner run` picks up the current window and commits inside it. A submission
for round *N* must land before round *N*'s close block; anything later competes
in round *N+1*.

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

## 8b · Running unattended

`mt miner serve` trains and submits without anyone typing a command. It reports
each epoch and each transition to the coordinator, then packages, uploads, logs
provenance and commits on chain exactly as `mt miner ship` does.

```bash
mt miner serve --train mytraining:run --epochs 40
```

`--train` names a `module:function` in your own code. The daemon calls it with a
hook and owns nothing else about your training loop:

```python
def run(hook):
    for epoch in range(40):
        loss = train_one_epoch()
        hook.on_epoch_end(epoch, {"loss": loss, "throughput": samples_per_second})
```

Two adapters ship for the common cases:

```python
from microtensor.miner.adapters import LoopAdapter, TrainerCallback

adapter = LoopAdapter(hook, samples_per_epoch=len(dataset))   # a plain loop
trainer = Trainer(..., callbacks=[TrainerCallback(hook)])     # HuggingFace
```

The state machine is `registered -> training -> packaging -> uploading ->
committing -> submitted`, with `failed` reachable from any of them. The chain
commitment stays yours throughout: it is a signed extrinsic, so only your hotkey
can produce one.

Pass `--no-telemetry` to submit unattended while reporting nothing.

### What telemetry is, and what it is not

Everything the daemon reports is observational. It never enters a certificate,
never affects a score, and never gates admission. Validators measure your
artifacts themselves on their own certified hardware, and that measurement is
the only figure that carries weight.

Hardware is self reported and labelled as such wherever it appears. Claiming an
accelerator you do not have changes nothing about admission, scoring, or
eligibility.

### Liveness

A miner silent for 300 blocks, about an hour, is dropped from the round.

The rule has one exception, and it matters:

| | |
|---|---|
| no commitment, silent past the threshold | dropped from the round |
| commitment on chain | measured, regardless of silence |

If you finished training and committed, your artifact is in a store any
validator can fetch. Losing your box after that does not cost you the round.

### The manual path still works

`package`, `provenance`, `ship` by hand remain supported. If you would rather
train separately and submit yourself, nothing here changes that.

### A public port

`mt miner serve` also answers on-demand status queries over a Bittensor axon, so
a dashboard or an operator can look into one run. That needs a reachable IP and
port 8091 open.

---

## 9b · Releases, and why the cutoff matters

Rounds settle every day. Every 30 rounds the frontier is frozen and published as
a version, like `microtensor-code-mt3g-v1`. That version is what the API serves
and what customers actually download and deploy.

**On the frontier when the cycle's final round settles means your system ships
in that release.** The cutoff is that round's close block, the same deadline you
already work to, so nothing new to track except which round ends the cycle.

A release pays nothing. Emissions are settled per round exactly as before, and a
boundary round pays what any other round would. Holding an improvement back to
land it just before a cutoff gains you nothing and costs you the rounds you sat
out.

`mt miner status` shows where you stand:

```
competition code/mt-3g
round       47
hotkey      5F3sa2TJ...
files       3  (1.21 GiB)
signed      yes
declared    size 1288490188, rss 2952790016, p95 165ms

release     microtensor-code-mt3g-v2
cutoff      13 round(s), block 4180000
frontier    yes, rank 2 of 7 by cost
measured    quality 0.8140, 118.0 ms
contribution front 0.62 · router 0.28 · specialist 0.10
milestone   0.85 quality under 150.0 ms  ·  unmet
```

The contribution line is the one to act on. It decomposes what your system
earned across the three components you declared, so it tells you which part is
carrying the result and which is not worth its cost. A specialist at 0.10 is a
specialist worth cutting or retraining.

Pass `--offline` to print the local half without asking the API.

---

## 10 · What gets you zeroed

| | |
|---|---|
| any component declares a base model off the allowlist | rejected at discovery, reason names the component |
| any component has no training run | rejected at discovery, reason names the component |
| router reads a feature that is not on the allowlist | rejected at discovery |
| router graph contains a disallowed operator | rejected at discovery |
| router artifact is over 4 MiB | rejected at discovery |
| specialist placed anywhere but the host profile | rejected at discovery |
| the same component digest appears twice | rejected at discovery |
| a router without a specialist, or a specialist without a router | rejected at discovery |
| no training run for your hotkey | rejected at discovery |
| training run digest does not match the artifact | rejected at discovery |
| training run declares a different track, class or base model | rejected at discovery |
| training run finished after you committed | rejected at discovery |
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

## 11 · Where the wins are

- **Quantise past where it looks safe.** The gate is on the envelope, not on
  parameter count. int8 or int4 that holds accuracy beats fp16 that misses RSS.
- **Peak RSS is measured at your declared maximum input, under sustained load.**
  not at load, not at steady state. For autoregressive models the KV cache grows
  linearly in context. Declare a maximum you can actually hold, then hold it.
- **Cold start is measured on an unwarmed cache.** A single warm measurement is
  precisely what a miner would construct if permitted to, so it is not admissible.
- **There is no ranked list.** A system that nothing else beats on both
  quality and cost sits on the frontier, and earns in proportion to the region
  it alone covers. Being second on accuracy costs you nothing if you are
  cheaper.
- **Find empty ground.** A system nobody is near earns well even when
  something else is more accurate. A system sitting on top of an existing one
  earns almost nothing, because the ground it covers was already covered.
- **Landing beside a leader earns nothing.** Two systems inside one epsilon
  neighbourhood collapse to the earlier commitment, so copying a frontier
  system and shaving it slightly leaves you with no position at all.
- **Incumbency decays.** Holding a frontier position across rounds with no
  improvement in either coordinate bleeds share to systems that moved. Ship
  improvements.
- **Each component is paid separately.** Your front, router and specialist are
  priced by the measured difference each makes against the published role
  baseline, so a component that adds nothing is visibly worth nothing.

---

## 12 · Submitting a system

This is the live path. A system is three parts. A **front** bound to the class ceiling, running on every
query. A **router** deciding which answers to keep. A **specialist** on the host
profile, answering only what escalates. Declare them in `system.json` beside your
artifact:

```json
{
  "schema_version": 1,
  "front":      {"role": "front",      "artifact_digest": "sha256:...", "placement": "mt-3g",  "path": "front"},
  "router":     {"role": "router",     "artifact_digest": "sha256:...", "placement": "mt-3g",  "path": "router.json"},
  "specialist": {"role": "specialist", "artifact_digest": "sha256:...", "placement": "mt-16g", "path": "specialist"},
  "router_features": ["seq_logprob_norm", "schema_valid"]
}
```

A router is data, not code. It is a threshold table or a small ONNX graph over
the published feature list, and the validator interprets it. You cannot ship a
routing function, and features you compute yourself never reach the decision:
the validator derives all of them from what your front emitted.

Tune it locally before you submit:

```bash
mt miner simulate --corpus ./corpus --limit 200
```

That runs the whole cascade over the public training split and reports resolve
rate, expected cost per query, end-to-end quality, and the uplift escalation
bought you. If the uplift is zero or negative, a router that never escalates
would score the same for less, and the frontier will price it accordingly.

Three things are worth knowing before you spend a week on this:

**Calibration beats accuracy.** The router can only act on what the front
exposes. A front that is accurate but confidently wrong gives the router nothing
to separate, so its errors pass through and end-to-end quality collapses. A
slightly less accurate front whose confidence orders its right and wrong answers
well escalates close to exactly what it would have failed, and wins on both
axes. Only the ordering matters, not the absolute value, since a monotone
transformation is absorbed by the threshold.

**Escalating everything is not a strategy.** The specialist's cost enters your
expected cost weighted by how often you escalate. Route everything and you carry
the specialist's full cost and sit at the expensive end of the frontier.

**Cheaper at lower quality still earns.** Emission follows exclusive
hypervolume, so a system that opens a genuinely new trade-off point is paid for
what it uniquely adds, whether that is the best quality anyone reached or the
cheapest anyone reached at usable quality.

---

## 13 · Starting simple

**A front alone is a valid system.** Submit one component, leave the router and
specialist null, and you compete. You will sit at the cheap end of the frontier,
where cost is lowest and the ground is often empty, and you will earn there.

```bash
mt miner init --track code --class mt-3g --front ./front
mt miner selfcheck
mt miner package
mt miner ship
```

This is the recommended first submission. It gets you a measured position, a
certificate, and a real number to improve against, without solving routing and
escalation on day one.

Adding a router and specialist buys quality and pays for it in escalation cost,
which moves you up and to the right. Whether that trade is worth making is
exactly what the frontier measures, and you are better placed to judge it once
you can see where your front actually landed.
