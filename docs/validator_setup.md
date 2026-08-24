# Running a Microtensor validator

Netuid **92** on finney.

## Requirements

| | |
|---|---|
| OS | Linux (the sandbox needs `resource.setrlimit`) |
| CPU | 8 cores |
| RAM | 32 GB |
| Disk | NVMe. 200 GB to start, 1 TB comfortable for a long run |
| GPU | none |
| Network | 1 Gbps, 500 GB to 1 TB monthly transfer |
| Uptime | continuous. Weights are submitted every 300 blocks (about 60 min); a round runs 21,600 blocks (about 3 days) and submissions close 7,200 blocks before it ends |
| Credentials | any W&B account (`wandb login`); no key is issued by us |

## 1. Install

```bash
git clone https://github.com/microtensor-io/microtensor-subnet
cd microtensor-subnet

python -m venv .venv && source .venv/bin/activate
pip install -e ".[validator,gguf,huggingface,s3]"
```

Install with `-e`. A plain install copies the package into site-packages,
and every later `git pull` silently changes nothing while the service keeps
running old code.

`validator` carries the ONNX engine; `gguf` adds the llama.cpp one. Install
the formats the arenas you measure accept. A format you skip simply does not
register, and `mt inspect engines` shows what a build will run.

`huggingface` and `s3` are fetch backends. Install the ones matching the source
schemes miners use. `https` needs nothing extra.

## 2. Register

```bash
btcli subnet register --netuid 92 --wallet.name <coldkey> --wallet.hotkey <hotkey>
btcli stake add --netuid 92 --wallet.name <coldkey> --amount <alpha>
```

Weights are only counted if your hotkey holds a validator permit.

## 3. Get authorized for coordinated rounds

The chain permit lets you vote. Taking assignments in coordinated rounds
additionally requires the operator to authorize your hotkey on the control
plane. Send your hotkey to the operator and wait for confirmation before
expecting work.

Without this you start cleanly, verify the config, adopt and verify the
coordinator's weights, and never measure anything. The validator logs
`no assignment document` at round close; that is this, not a fault.

Assignments are drawn when a round opens. A validator authorized mid round
receives its first assignment at the next round open.

## 4. Configure

```bash
export MT_NETWORK=finney
export MT_WALLET_NAME=<coldkey>
export MT_WALLET_HOTKEY=<hotkey>
export MT_COORDINATOR_URL=https://coordinator.microtensor.cloud
```

`MT_NETUID` defaults to 92. Flags override environment; `--coordinator` and
`MT_COORDINATOR_URL` are the same setting.

W&B credentials are required, from **any** W&B account; we issue nothing:

```bash
wandb login
```

`wandb login` writes `~/.netrc` and that is enough; `WANDB_API_KEY` works too.
Every submission carries a public training run in `microtensor/training-runs`
bound to its artifact digest, and a validator that cannot read that project
admits nobody. The project is world-readable, so any valid key can.

## 5. Corpus

A coordinated validator takes the corpus from the coordinator. You supply
nothing. Every worker in a round measures the same tasks, and reports carry a
content digest of what the worker actually holds, so a mismatch is rejected
rather than folded into the majority as a disagreement.

Only a standalone validator supplies its own, one `<track>.jsonl` per track
under `$MT_HOME/corpus`, one task per line:

```json
{"ref": "code-0001", "prompt": "…", "gold": {"cases": [true, true]}, "partition": "rotating", "max_output_tokens": 512}
```

`partition` is `rotating` or `fixed`, and every corpus must carry a fixed
partition. Per round the validator draws `⌈0.7·N⌉` rotating tasks with the round
seed and takes `⌊0.3·N⌋` fixed tasks unchanged.

## 6. Certify the host

```bash
mt validator certify mt-3g --cooling-mode active --power-mode performance
```

Cooling and power modes are pinned into the device profile hash. Changing either
one later means certifying again.

## 7. Verify

```bash
mt inspect engines      # must show: sandbox enforced
mt validator status
mt inspect tracks
```

Then test the full loop on synthetic data, with no chain involved:

```bash
mt validator loopback --rounds 3 --miners 4
```

Then a real round without submitting:

```bash
mt validator once --dry-run
```

## 8. Run

```bash
mt validator run
```

Coordinated (measures an assigned subset, adopts a settlement it recomputes):

```bash
mt validator run --coordinator https://coordinator.microtensor.cloud --auto-update
```

A coordinated validator takes the open round from the coordinator, because
the operator opens rounds. The block schedule is only the fallback when the
coordinator cannot be reached. Between rounds it idles on
`round N is already settled; waiting for the next`, which is correct.

To run standalone, leave `--coordinator` unset and `MT_COORDINATOR_URL` empty.
`--auto-update` installs signed releases between rounds and exits for the
supervisor to restart it, so only use it under systemd or docker. Backgrounded
in a shell, the first update leaves it down.

## What the logs show

Startup, in order; each line is a stage completing:

```
training run store reachable at microtensor/training-runs
Enabling default logging (Warning level)        ← bittensor, during wallet load
taking assignments from the coordinator at …    ← metagraph fetched, hotkey checked
validator up on netuid 92 across N competitions
round 1236 open: 5400 blocks until submissions close
```

The metagraph fetch between the second and third lines takes a minute or two
and prints nothing. It is not stuck.

**The long wait is normal.** A round is 21,600 blocks, about 3 days, and a
validator acts when the round closes, not before. Until then it repeats
`round N open: M blocks until submissions close` every few minutes, refreshing
weights on the way. A silent process here is a broken one; a chatty one
counting down blocks is healthy.

Before an arena is live there is nothing to measure, and the round settles on
the reserved hold. The validator fetches that settlement, recomputes it,
checks the reserved uid against its own metagraph, and submits the same
weights the coordinator would: verified rather than relayed. You still set
weights every round from day one.

`no corpus served yet; waiting for an arena to go live` is likewise a wait,
not a fault, and is rechecked every round.

Versions before 0.1.10 lost every log line after wallet load; bittensor
resets the root logger to WARNING. Upgrade rather than debug the silence.

### systemd

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
ExecStart=/opt/microtensor/.venv/bin/mt validator run --coordinator https://coordinator.microtensor.cloud --auto-update
Restart=always
RestartSec=30
TimeoutStopSec=1800
SuccessExitStatus=75

[Install]
WantedBy=multi-user.target
```

`TimeoutStopSec` is long so `SIGTERM` lets the current round finish.

The unit's `User` must be the account that ran `wandb login`, or set
`Environment=WANDB_API_KEY=<key>` explicitly. A service user without either
fails at boot naming the run store.

### Docker

```bash
cd deploy
MT_WALLET_NAME=<coldkey> MT_WALLET_HOTKEY=<hotkey> \
  docker compose -f docker-compose.validator.yml up -d
```

## Updates

Validators on different builds compute different weights, and `version_key` is
derived from `MECHANISM_VERSION`, so drift splits consensus.

Auto-update is off by default:

```bash
mt validator run --auto-update --signing-key <pinned-ed25519-pubkey>
```

| situation | result |
|---|---|
| patch or minor, same mechanism | installed, exit 75, supervisor restarts |
| evaluation in flight | deferred to the next submission window |
| mechanism change, no activation block | held |
| mechanism change with an activation block | deferred to that block, then held unless `--allow-mechanism-change` |
| major version bump | held |
| unsigned, missing SHA256SUMS, or digest mismatch | refused |

Exit code 75 means restart me. Docker's `restart: unless-stopped` covers it.

```bash
mt update check             # what would happen now, and why
mt update list              # published releases
mt update apply --dry-run   # verify without installing
```

`mt update check` exits 2 when something is waiting for you.

## Abstention

A fault of the artifact scores zero. A fault of your infrastructure abstains.

| event | result |
|---|---|
| over a class ceiling, or over its own declaration | inadmissible, scores 0 |
| task times out, produces nothing, or the worker dies | that task scores 0 |
| artifact unfetchable after retries | abstain |
| no engine available | abstain |
| run store unreachable after retries | abstain |
| fewer than 50% of submissions scored | abstain |

Abstaining sets no weights that round and leaves EMA state untouched. A partial
vector is never submitted, because a missing track removes that track's whole
emission share.

A round that evaluates cleanly but pays nobody, because no artifact has reached
`MIN_ROUNDS_OBSERVED` yet, is settled rather than abstained.

## The dropped set

Settlements carry a `dropped` map of miners that went quiet during training
without committing, each with the block they were last heard from. It is
verified the same way the advisory set is: recomputed from the published
settlement, and a mismatch rejects the settlement.

A commitment ends the check. A miner that committed on chain is measured whether
or not it kept reporting, because the artifact is already fetchable.

Standalone validators see no telemetry and are unaffected. They enrol from chain
commitments alone, which is correct precisely because a commitment always
survives silence.

---

## Weights and rounds

Weight submission is not tied to round completion. The validator re-submits its
standing vector every 300 blocks, so it is never silent between rounds, and a
round runs about 3 days. A validator that only set weights when a round
finished would go quiet for hundreds of epochs and give up the dividends to
whoever kept submitting.

The refresh republishes what the last round settled. It recomputes nothing.

## Round phases

```
[start ─────────────────── close) [close ──────── deadline] [·· margin ··]
 submissions open           freeze  evaluate + settle          extrinsic slack
```

1. Wait for the close block.
2. Seed from `hash(B_close)`.
3. Discover: read commitments, verify signatures and digests.
4. Materialise: fetch artifacts, re-derive file tree digests before executing.
5. Profile: size, peak RSS, TTFT p50/p95, cold start, inside the jail.
6. Gate: over a ceiling or over its own declaration means it does not exist this round.
7. Score: run the task set in the jail, quantised to 4 decimals.
8. Settle: rank with hysteresis, apply incumbent decay and the concentration cap,
   blend into the prior vector, submit.

Each admitted system runs one cascade over the round's task set. Frontier members
get one further run per declared component. Component outputs are cached by digest
within the round.

## Troubleshooting

| symptom | cause |
|---|---|
| `mt inspect engines` shows `UNAVAILABLE` | not on a host that can validate |
| refuses to start, CPU limit does not bind | kernel will not enforce budgets, pass `--allow-degraded` to run abstain only |
| fails at boot naming the run store | `WANDB_API_KEY` missing, wrong, or the API is unreachable |
| `mt validator status` warns about the permit | hotkey holds no validator permit |
| holds every round with `no assignment document` | the operator has not authorized your hotkey for coordinated rounds |
| starts, adopts weights, never measures | authorized mid round; assignments arrive at the next round open |
| `git pull` changes nothing | the install was not editable; run `pip install -e .` once and restart |
