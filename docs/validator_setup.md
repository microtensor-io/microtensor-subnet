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
| Uptime | continuous. Weights are submitted every 300 blocks (about 60 min); a full measurement round completes every 7200 blocks (about 24 h) |
| Credentials | a W&B read key for `microtensor/training-runs` |

## 1. Install

```bash
git clone https://github.com/microtensor-io/microtensor-subnet
cd microtensor-subnet

python -m venv .venv && source .venv/bin/activate
pip install ".[validator,huggingface,s3]"
```

`huggingface` and `s3` are fetch backends. Install the ones matching the source
schemes miners use. `https` needs nothing extra.

## 2. Register

```bash
btcli subnet register --netuid 92 --wallet.name <coldkey> --wallet.hotkey <hotkey>
btcli stake add --netuid 92 --wallet.name <coldkey> --amount <alpha>
```

Weights are only counted if your hotkey holds a validator permit.

## 3. Configure

```bash
export MT_NETWORK=finney
export MT_WALLET_NAME=<coldkey>
export MT_WALLET_HOTKEY=<hotkey>
export MT_HOME=~/.microtensor
export WANDB_API_KEY=<read key>
```

`MT_NETUID` defaults to 92. The coordinator and control plane default to
`https://coordinator.microtensor.cloud` and `https://api.microtensor.cloud`;
override with `MT_COORDINATOR_URL` and `MT_SERVER_URL`.

`WANDB_API_KEY` is required. Every submission carries a public training run in
`microtensor/training-runs` bound to its artifact digest, and a validator that
cannot read that project admits nobody. A read key is enough.

## 4. Add a corpus

One `<track>.jsonl` per track under `$MT_HOME/corpus`:

```
~/.microtensor/corpus/
├── code.jsonl
├── document.jsonl
├── analytics.jsonl
└── support.jsonl
```

One task per line:

```json
{"ref": "code-0001", "prompt": "…", "gold": {"cases": [true, true]}, "partition": "rotating", "max_output_tokens": 512}
```

`partition` is `rotating` or `fixed`. Every corpus must carry a fixed partition
or the loader refuses it. Per round the validator draws `⌈0.7·N⌉` rotating tasks
with the round seed and takes `⌊0.3·N⌋` fixed tasks unchanged.

## 5. Certify the host

```bash
mt validator certify mt-3g --cooling-mode active --power-mode performance
```

Cooling and power modes are pinned into the device profile hash. Changing either
one later means certifying again.

## 6. Verify

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

## 7. Run

```bash
mt validator run
```

Coordinated (measures an assigned subset, adopts a settlement it recomputes):

```bash
mt validator run --coordinator https://coordinator.microtensor.cloud
```

Pass `--standalone` to ignore a configured coordinator.

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
ExecStart=/opt/microtensor/.venv/bin/mt validator run
Restart=always
RestartSec=30
TimeoutStopSec=1800
SuccessExitStatus=75

[Install]
WantedBy=multi-user.target
```

`TimeoutStopSec` is long so `SIGTERM` lets the current round finish.

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

## Weights and rounds

Weight submission is not tied to round completion. The validator re-submits its
standing vector every 300 blocks, so it is never silent between rounds. A round
takes about 24 h because a full measurement pass over every submitted system
does; a validator that only set weights when a round finished would go quiet for
twenty epochs at a time and give up the dividends to whoever kept submitting.

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
