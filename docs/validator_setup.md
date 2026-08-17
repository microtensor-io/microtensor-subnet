# Running a Microtensor validator

A validator does all the work in this subnet. It reads commitments, fetches every
artifact, measures each one's deployment envelope on certified hardware, executes
it against a chain-seeded task set inside a resource jail, ranks the survivors,
and publishes one weight vector per round.

Miners run none of this. If you are here to submit a model, read
[miner_setup.md](miner_setup.md) instead. The two roles share almost no surface.

---

## 1 · What you are signing up for

Unlike a miner's rig, this is enforced. Envelope measurements are the product of
this subnet, so a measurement taken on the wrong device is excluded from
aggregation and a measurement taken without limits is not evidence at all.

| | |
|---|---|
| **OS** | Linux. Not optional, the jail needs `resource.setrlimit` |
| **CPU** | 8 cores |
| **RAM** | 32 GB |
| **Disk** | 1 to 2 TB NVMe |
| **GPU** | **none** |
| **Network** | 1 Gbps, 500 GB to 1 TB monthly transfer |
| **Uptime** | continuous, runs a neuron, one round every `ROUND_BLOCKS` (7200 blocks ≈ 24 h) |
| **Bench** | one reference device for the launch class, `mt-3g` |
| **Credentials** | a W&B read key for `microtensor/training-runs` |
| **Fails** | closed, so an unenforceable sandbox refuses to run rather than guess |

**No GPU, deliberately.** The engine runs on `CPUExecutionProvider`, so
validating is cheap and is not a capital barrier. The transfer budget is driven
by artifact fetches: a full round pulls every submission it has not cached.

Full breakdown in [min_compute.yml](../min_compute.yml).

The hard requirement is the sandbox. Envelope measurements are the product of
this subnet, and a measurement taken without CPU and memory limits is not
evidence. `mt validator run` refuses to start on a host where
`resource.setrlimit` is unavailable, and the jail records `sandboxed=False` on
any run that slipped through, so an unenforced measurement can never reach a
weight vector.

Confirm before anything else:

```bash
mt inspect engines
```

`sandbox   enforced` and at least one engine must appear. If it says
`UNAVAILABLE`, you are not on a host that can validate.

---

## 2 · Reference hardware

Every class publishes a `device_profile`. A validator whose hardware does not
conform **may still score accuracy**, since accuracy is hardware-independent, but its
envelope measurements are excluded from aggregation, and conforming measurements
aggregate by median.

| class | size ceiling | RSS ceiling | p95 TTFT | reference |
|---|---|---|---|---|
| `mt-16g` | 8 GiB | 16 GiB | 400 ms | x86-64 server, no accelerator |
| `mt-4g` | 2.5 GiB | 4 GiB | 120 ms | consumer or embedded GPU |
| `mt-3g` | 1.5 GiB | 3 GiB | 180 ms | developer workstation |
| `mt-1g` | 600 MiB | 1 GiB | 300 ms | mobile SoC or NPU |

Check what your host reports:

```bash
python -c "from microtensor.envelope.device import detect; print(detect().digest, detect().to_dict())"
```

Run one validator process per class you can certify. A single host that
genuinely matches `mt-16g` should not also claim `mt-1g`.

---

## 3 · Install

```bash
git clone https://github.com/microtensor-io/microtensor-subnet
cd microtensor-subnet

python -m venv .venv && source .venv/bin/activate
pip install ".[validator,huggingface,s3]"
```

`huggingface` and `s3` are fetch backends. Install the ones matching the source
schemes miners actually use; `https` needs nothing extra.

### Register the hotkey

Microtensor is **netuid 92** on finney, and that is the built-in default.

```bash
btcli subnet register --netuid 92 --wallet.name <coldkey> --wallet.hotkey <hotkey>
btcli stake add --netuid 92 --wallet.name <coldkey> --amount <alpha>
```

You need a validator permit to have weights counted. `mt validator status` warns
if your hotkey holds none.

---

## 4 · The corpus

**A validator without a corpus scores nothing.** Place one `<track>.jsonl` per
track you serve under `$MT_HOME/corpus`:

```
~/.microtensor/corpus/
├── code.jsonl
├── document.jsonl
├── analytics.jsonl
└── support.jsonl
```

Each line is one task:

```json
{"ref": "code-0001", "prompt": "…", "gold": {"cases": [true, true]}, "partition": "rotating", "max_output_tokens": 512}
```

`partition` is `rotating` or `fixed`. Every corpus **must** carry a fixed
partition. The loader refuses one without it, because a corpus with no
invariant slice cannot tell "the models improved" from "the generator drew an
easier batch".

Per round the validator draws `⌈0.7·N⌉` rotating tasks with the round seed and
takes `⌊0.3·N⌋` fixed tasks unchanged. Both must come from distributions absent
from public corpora, or you are measuring memorisation.

---

## 5 · Configure

Environment variables are read with an `MT_` prefix; flags override them.

```bash
export MT_NETWORK=finney
export MT_WALLET_NAME=<coldkey>
export MT_WALLET_HOTKEY=<hotkey>
export MT_HOME=~/.microtensor
```

`MT_NETUID` defaults to 92, so you do not normally set it.

Verify before running for real:

```bash
mt validator status
mt inspect tracks
```

---

## 6 · Run

Certify the host for each class you serve. This runs a fixed, versioned
workload, pins your declared thermal and power policy into the device
profile hash, and records the measurements:

```bash
mt validator certify mt-3g --cooling-mode active --power-mode performance
mt validator certify mt-4g
```

A policy change is a different profile, so changing cooling or power modes
after certifying means re-certifying. Until tolerance bands are published
for the launch classes, certify records your numbers for calibration.

At startup the validator also probes whether the CPU limit actually binds by
running a spinner under a one-second budget. If the kernel never kills it,
budgets cannot be enforced and slow infrastructure would be misattributed as
artifact fault, so the validator refuses to start; pass `--allow-degraded`
to run abstain-only instead (it discovers, fetches and logs, but sets no
weights).

Prove the machinery works before you touch a chain at all. Loopback stands up a
synthetic chain, a seeded corpus and several fake miners, then runs real rounds
end to end on your own hardware:

```bash
mt validator loopback --rounds 3 --miners 4
```

```
   round  status      participants  scored  weights  reason
       3  settled                4       4        0  evaluated cleanly, but no artifact is eligible for emission yet
       4  settled                4       4        4
       5  settled                4       4        4

weight vectors submitted: 2
  uids [1, 2, 3, 4]  values [20565, 17481, 14859, 12630]  sum 65535
```

Round 3 paying nobody is correct: an artifact must be observed for
`MIN_ROUNDS_OBSERVED` rounds before it can earn. The values are the geometric
curve at decay 0.85, and they sum to exactly 65535.

Then dry run against the real chain, which evaluates a full round and computes
weights without submitting them:

```bash
mt validator once --dry-run
```

Then take it live:

```bash
mt validator run
```

Under systemd:

```ini
[Unit]
Description=Microtensor validator
After=network-online.target

[Service]
Type=simple
User=validator
Environment=MT_NETUID=92
Environment=MT_WALLET_NAME=<coldkey>
Environment=MT_WALLET_HOTKEY=<hotkey>
Environment=MT_HOME=/var/lib/microtensor
ExecStart=/opt/microtensor/.venv/bin/mt validator run
Restart=always
RestartSec=30
TimeoutStopSec=1800

[Install]
WantedBy=multi-user.target
```

`TimeoutStopSec` is long on purpose. `SIGTERM` asks the loop to finish the
current round rather than abandon a half-scored one.

Or with Docker:

```bash
cd deploy
MT_WALLET_NAME=<coldkey> MT_WALLET_HOTKEY=<hotkey> \
  docker compose -f docker-compose.validator.yml up -d
```

---

## 7 · The round, phase by phase

```
[start ─────────────────── close) [close ──────── deadline] [·· margin ··]
 submissions open           freeze  evaluate + settle          extrinsic slack
```

1. **Wait** until the close block. Submissions are open until then.
2. **Seed** from `hash(B_close)`. Nobody who submitted could have seen it.
3. **Discover.** Read commitments, fetch each manifest, verify its signature and
   that it hashes to the committed digest. Malformed pointers are skipped.
4. **Materialise.** Fetch every artifact and re-derive its whole file tree
   digest *before* any execution begins, so fetch failures surface at round start.
5. **Profile.** Measure size, peak RSS at declared maximum input under sustained
   load, TTFT p50/p95, cold start. Inside the jail.
6. **Gate.** Binary. Over a class ceiling, or over its own declaration, and the
   artifact does not exist for the round.
7. **Score.** Run the task set in the jail, quantising to 4 decimals so two
   validators computing in different float orders emit identical vectors.
8. **Settle.** Rank with hysteresis, apply incumbent decay and the concentration
   cap, blend into the prior vector asymmetrically, submit.

---

## 8 · Abstention

**A fault of the artifact scores zero. A fault of your infrastructure abstains.**

| event | consequence |
|---|---|
| over a class ceiling, or over its own declaration | inadmissible, scores 0 |
| task times out, produces nothing, or the worker dies | that task scores 0 |
| artifact unfetchable after retries | **abstain** |
| no engine available | **abstain** |
| fewer than 50 % of submissions scored | **abstain** |

Abstaining sets no weights that round. EMA state is untouched and you resume
next round. A partial vector is never submitted: since every validator scores
every track, a missing track removes that track's whole emission share, which is
a consensus divergence rather than a smaller sample.

A round that evaluates cleanly but pays nobody, because no artifact has been
observed for `MIN_ROUNDS_OBSERVED` rounds yet, is **settled**, not abstained.
The evaluations are real and next round depends on them.

---

## 9 · Staying on the right version

Validators that run different builds compute different weights. Worse, weight
`version_key` is derived from `MECHANISM_VERSION`, so a validator on a newer
mechanism submits a *different* version key, and the chain treats it as a different
mechanism. Version drift is a consensus problem here, not a hygiene problem.

Auto-update is **off by default**. Arm it explicitly:

```bash
mt validator run --auto-update --signing-key <pinned-ed25519-pubkey>
```

What it will and will not do:

| situation | what happens |
|---|---|
| patch or minor release, same mechanism | installed, validator exits 75, supervisor restarts it |
| round is sealed (evaluation in flight) | **deferred** until the next submission window |
| release changes `MECHANISM_VERSION`, no activation block | **held**, since applying at a different moment than other validators splits consensus |
| release changes the mechanism *with* an activation block | deferred until that block, then held unless `--allow-mechanism-change` |
| major version bump | **held**, never automatic |
| SHA256SUMS missing, unsigned, or the digest mismatches | refused, validator stays on the running build |
| `pip install` fails | logged, no restart, validator keeps running the old build |

The restart never lands mid-evaluation. The check only fires while submissions
are open, which is a ~22 hour window per round, so there is no urgency and no
risk of abandoning a half-scored round.

Inspect without applying:

```bash
mt update check       # what this validator would do right now, and why
mt update list        # published releases on a channel
mt update apply --dry-run   # download and verify, install nothing
```

**Exit code 75 means "restart me".** Wire your supervisor for it:

```ini
[Service]
ExecStart=/opt/microtensor/.venv/bin/mt validator run --auto-update
Restart=always
RestartSec=15
SuccessExitStatus=75
```

Docker's `restart: unless-stopped` already handles it. Under PM2 use
`--exp-backoff-restart-delay=15000`.

If you would rather not run unattended updates, leave the flag off and watch
`mt update check` from cron. It exits 2 when something is waiting for you.

---

## 10 · Operating

```bash
mt validator status            # local state, cache size, last settled round
mt inspect rounds --limit 20   # recent rounds and why any abstained
mt inspect hotkey <ss58>       # one miner's history: admitted, score, gate reason
mt inspect engines             # what this host can execute, sandbox status
```

State lives in `$MT_HOME/state/validator.sqlite` (WAL). It is the only thing you
must back up. The artifact cache and work directory are both disposable.

**Watch for**
- repeated `abstaining:` lines, usually one unreachable source or a missing engine
- cache thrash, so raise `--cache-cap-bytes` if eviction runs every round
- `holds no validator permit`, meaning your weights are being ignored

**Upgrades.** The schema carries a `user_version` and refuses to open state
written by a newer build. Upgrade the validator, never downgrade its state.

**Keys.** Only the hotkey is needed to run. Keep the coldkey off the machine.
