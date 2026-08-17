# Microtensor

**Small specialist models, verified to fit the hardware they claim.**
Bittensor subnet 92.

Microtensor rewards miners for compressing frontier models into small
specialists that provably fit a hardware envelope. It runs as a continuous
benchmark:

1. **Miners** compress a specialist from the pinned base models against the
   published corpus and reference completions, upload the artifact, and
   commit a 128-byte pointer on chain before the round closes.
2. **Validators** fetch each artifact and run it themselves: they measure
   size, sustained peak memory, and p95 latency on certified reference
   hardware, gate on the class ceiling and the miner's own declaration, and
   score survivors on execution pass rate against a chain-seeded hidden task
   set.
3. **The chain** aggregates validator weights through Yuma Consensus. The top
   8 per competition earn on a geometric curve.

Miners run no axon, serve nothing, and need no uptime. Validators need Linux,
no GPU, and one reference device per class they certify.

[Mine](docs/miner_setup.md) · [Validate](docs/validator_setup.md) ·
[Mechanism](docs/mechanism.md)

---

## Live competitions

| Track | Class | Size | Peak RSS | p95 first output | Emission |
|---|---|---|---|---|---|
| `code` | `laptop` | 1.5 GiB | 3 GiB | 180 ms | 60 % |
| `code` | `edge-gpu` | 2.5 GiB | 4 GiB | 120 ms | 40 % |

**Metric:** execution pass rate. Generated code runs against hidden tests in
a sandbox; the score is the fraction that pass. Computed, never judged.

**Next track:** `detect` (vision). The registry in
[`tracks.py`](microtensor/core/tracks.py) carries the full multimodal set as
disabled stubs; tracks are enabled by governance as corpora and reference
benches come online.

## Constraints

| | |
|---|---|
| Base models | Pinned allowlist (Qwen3 / Llama 3.2, 0.6B–4B) in [`constants.py`](microtensor/core/constants.py) — exact HF revisions, published at corpus freeze |
| Corpus | Public train split + prompts published per version; hidden tests and the rotating draw are validator-side |
| Reference completions | One vetted completion per train-split task ships with each corpus release — miners train on prompt/completion pairs without running a large model |
| Baseline | `mt-code-baseline` and its score are published in [releases](../../releases); that is the number to beat |
| Formats | safetensors, GGUF, ONNX |
| Submission | One per competition per round; chain rate-limits commits to one per 20 minutes |
| Artifact hosting | `hf:` (repo@commit-sha), `ipfs:`, `s3:`, `r2:`, `https:` — digest-verified after fetch |

## Rounds

A round is 7 200 blocks (≈ 24 h). Submissions close 600 blocks before the
round ends; the task set is seeded from the hash of that close block, so
nobody can see the questions while submissions are open, and anyone can
reproduce the draw after. Your artifact must be fetchable by validators at
the close block.

## Getting started

```bash
pip install -e .
mt inspect tracks          # live competitions, ceilings, emission shares
mt inspect engines         # what this host can execute; whether the jail binds

# Miner: four commands, no uptime
mt miner init --track code --class laptop --artifact ./my-model --source hf:you/mt-code@<sha>
mt miner selfcheck         # measure your own envelope before declaring it
mt miner package
mt miner ship              # upload + commit on chain

# Validator: prove the machinery before touching a chain
mt validator loopback --rounds 3 --miners 4
mt validator certify laptop
mt validator run
```

Hardware guidance for miners and the enforced validator floor are in
[`min_compute.yml`](min_compute.yml). The full mechanism — what is frozen,
what is tunable, and why every rule exists — is
[`docs/mechanism.md`](docs/mechanism.md).

## Status

Subnet 92 on finney. The corpus version, the baseline score to beat, the
pinned base-model revisions, and the reference device profiles for both launch
classes are published in [releases](../../releases). Everything in this
repository runs today: `mt validator loopback` settles rounds end to end on
your own machine with no chain and no network.
