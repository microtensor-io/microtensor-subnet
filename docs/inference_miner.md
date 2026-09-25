# Running an inference miner

You serve certified systems on your own hardware and are paid for tokens
clients actually bought. One process, one outbound connection, many models.

```
 gateway ──websocket──► mt operator run ──http──► sglang :18080  (mt/invoice-4g front)
              ▲               │         ──http──► sglang :18081  (mt/invoice-4g specialist)
              │               │         ──http──► sglang :18082  (mt/text2sql-16g)
              │               └─ router in process, one small onnx classifier per system
              │               │
              │               └─► returns text + prompt/completion token ids
              │
 validator ───┘  samples a response, prefills it through its own copy of the
                 certified artifact, and judges the tokens
```

You dial out. No inbound port, no public address, no certificate, no axon.

To choose between the three kinds of miner read [miner_setup.md](miner_setup.md).

---

## 1 · Requirements

| | minimum | notes |
|---|---|---|
| OS | Linux, macOS or Windows | tested on Ubuntu 22.04 and 24.04 |
| Python | 3.10 | 3.11 or newer preferred |
| GPU | **required** | NVIDIA, AMD or Apple silicon; the agent refuses to start without one |
| VRAM | 8 GB | this is the budget the planner spends; see the table below |
| CPU | 8 cores | for the agent and the engines, not for generation |
| RAM | 16 GB | host memory, separate from the card |
| Disk | 50 GB | artifacts are 1 to 16 GiB each and cached by digest |
| Network | outbound 443 | nothing inbound |
| Engine | SGLang, plus llama.cpp for GGUF | the agent picks from the format the artifact declares |
| Collateral | 1 TAO | forfeited on a cheat verdict |

### A GPU is required, and here is why

The agent detects your card at startup and refuses to run without one. It looks
for `nvidia-smi`, then `rocm-smi`, then Apple silicon, and prints what it found.

The reason is a distinction that trips people up. The arena measures a system on
**one CPU thread with no GPU offload**, seeded and greedy, because two
validators have to produce the same number and determinism demands it. That is a
cost figure for ranking systems on the frontier. It is not a serving target.

For scale, the live `guard/mt-4g` system measures at 3.75 tokens a second with a
p95 of 57 seconds on that reference thread. No client waits 57 seconds.

Serving is judged against a separate envelope, published per model, set by what
a client will actually tolerate. Your time to first token and time per output
token are measured against it, and missing it forfeits the epoch. A CPU cannot
hold that pace for any class we certify.

### How much card

Your VRAM is the budget the planner spends. Each system needs its peak resident
memory plus room for the key value cache at your concurrency.

| class | card that holds one comfortably | holds several |
|---|---|---|
| `mt-1g` | 6 GB | 12 GB and up |
| `mt-3g` | 8 GB | 16 GB and up |
| `mt-4g` | 8 GB | 24 GB and up |
| `mt-16g` | 24 GB | 48 GB and up |

More VRAM means more systems held at once, which means more of the catalogue you
can answer for and more traffic routed to you. `mt operator plan` shows exactly
what your card holds before you commit to anything.

A 24 GB card holds roughly four `mt-4g` systems at once and is a sensible
starting point.

---

### Engines and formats

The agent reads the format from the artifact's own manifest and starts the
engine that serves it. You install the engines; it chooses between them.

| format | engine | why |
|---|---|---|
| `awq` | SGLang with `--quantization awq_marlin` | INT4 weights on the Marlin kernel, the fast path on Ampere and newer |
| `fp8`, `safetensors` | SGLang | its native path |
| `gguf` | llama.cpp | what llama.cpp is for |

**Radix prefix caching is why SGLang.** It is on by default, and shared prefixes
across requests are what agent and chat traffic looks like. The gateway routes a
conversation back to the operator that already holds its cache, and SGLang
reuses it instead of recomputing the prefix. That pairing is the single biggest
win available to you. Chunked prefill is on, CUDA graphs are on, and speculative
decoding is off, which is right for models this size.

### Several systems on one card

Each engine gets `--mem-fraction-static` set to its share of the card, computed
from how many systems the planner chose, with eight percent left for the driver
and fragmentation. A system with a specialist takes two shares, because the
front and the specialist are separate engines.

This is the part the networks doing this today do not attempt. They dedicate a
card, or four, to one large model. Our systems are one to sixteen gibibytes, so
packing is where the advantage is, and it is why VRAM decides what you earn.

---

## 2 · Install

```bash
git clone https://github.com/microtensor-io/microtensor-subnet
cd microtensor-subnet
pip install -e ".[serving]"

pip install "sglang[all]"
```

Add llama.cpp as well if you intend to hold GGUF systems:

```bash
apt install -y llama.cpp
```

The `serving` extra pulls `websockets` and `httpx`. Neither is imported until
you actually serve.

You also need an engine. Any llama.cpp build works:

```bash
# Ubuntu
apt install -y llama.cpp          # or build from source
llama-server --version
```

Check the CLI is on PATH:

```bash
mt operator --help
```

---

## 3 · Register

```bash
mt operator register --label "frankfurt-1"
```

Prints your state and what collateral this network asks for:

```json
{
  "hotkey": "5G...",
  "state": "registered",
  "collateral_required_tao": 1.0,
  "collateral_pay_to": "5G9ca9pgbCa63aJWbDAF3dNLhUtbFDYf1fnPMnByf9uyUNFh"
}
```

Use a hotkey you do not mine or validate with. Roles are measured, scored and
paid separately.

---

## 4 · Post collateral

Send the transfer, wait for it to be in a block, then report it:

```bash
btcli wallet transfer --dest <collateral_pay_to> --amount 1.0 \
  --subtensor.network test

mt operator collateral --reference 0x<extrinsic hash> --block <block>
```

The server reads that block and verifies the transfer before it counts. A
reference that is not in that block, a transfer to the wrong address, or an
amount below the minimum is refused with the reason.

| state after | means |
|---|---|
| `registered` | collateral not yet accepted |
| `collateralised` | accepted, ready to declare and be probed |
| `probing` | a validator has started testing you |
| `admitted` | eligible, the gateway may route to you |
| `withdrawn` | a validator pulled you on trust; prove yourself again |
| `forfeited` | collateral taken after a cheat verdict |

---

## 5 · See what the network would give you

You do not pick models. The catalogue and the demand signal decide, and a
planner on your machine ranks them against your hardware.

```bash
mt operator plan --slots 16
```

```json
{
  "hold": [
    {"model": "mt/invoice-4g", "artifact_digest": "sha256:9f2c...", "concurrency": 8},
    {"model": "mt/text2sql-16g", "artifact_digest": "sha256:41ab...", "concurrency": 8}
  ],
  "fetch": ["mt/invoice-4g", "mt/text2sql-16g"],
  "drop": [],
  "disk_bytes": 19327352832,
  "memory_bytes": 21474836480,
  "reasons": {
    "mt/guard-4g": "no memory left after the models above it"
  }
}
```

Nothing is fetched and nothing is started. Every model it passed over carries a
reason.

### Ranking

```
wanted = requests_last_hour / (operators_online + 1)
```

Highest `wanted` first. Ties break toward a model nobody is serving, then toward
the smaller artifact. A model you already hold has its score multiplied by 1.5,
so the set does not thrash when two are close.

Refused outright: a system with no published serving envelope, one that does not
fit in memory or on disk, one this machine already measured as too slow for its
envelope, and anything your policy excludes.

On the first cycle a system is unmeasured, so the planner tries it. The agent
benchmarks each engine as it comes up, drops anything slower than its envelope,
and remembers the figure so later cycles never pick it again.

### Narrowing it

```bash
mt operator plan --track invoice --class mt-4g --not mt/guard-4g --max-models 4
```

| flag | default | does |
|---|---|---|
| `--slots N` | 8 | requests at once across every model, shared out between them |
| `--disk-gb N` | free disk | ceiling on artifact storage |
| `--memory-gb N` | free memory | ceiling on resident memory |
| `--max-models N` | slots | how many systems to hold at once |
| `--track T` | all | only this track, repeatable |
| `--class C` | all | only this hardware class, repeatable |
| `--not M` | none | never hold this model, repeatable |

---

## 6 · Run

```bash
mt operator run --auto \
  --slots 16 \
  --artifacts /data/artifacts \
  --engine llama-server \
  --threads 4
```

In order it: reads the catalogue, plans, fetches each artifact and checks its
digest against the certificate, starts one engine per system on a port from
18080 upward, waits for each to answer `/health`, declares each model to the
server, dials the gateway, and answers. It rereads the catalogue every 300
seconds and redials when what it should hold changes.

| flag | default | does |
|---|---|---|
| `--auto` | off | let the network choose; without it you list models yourself |
| `--artifacts PATH` | `artifacts` | artifact cache, keyed by digest |
| `--engine BIN` | per backend | the binary to start; `llama-server` or `python` by default |
| `--threads N` | engine default | passed through to each engine |
| `--gpu-layers N` | -1 | llama.cpp only; layers offloaded to the card, -1 is all |
| `--worker NAME` | hostname derived | names this machine when several share one hotkey |
| `--review-seconds N` | 300 | how often to reread the catalogue |
| `--gateway URL` | `wss://api.microtensor.cloud/v1/operators/socket` | where to dial |

### Several machines, one hotkey

```bash
# box A
mt operator run --auto --slots 16 --worker rack-1
# box B
mt operator run --auto --slots 32 --worker rack-2
```

Each registers as its own worker. A repeat hello with the same hotkey and worker
name supersedes the earlier connection, so a restart never leaves a ghost.

### Choosing yourself instead

```bash
mt operator declare --model mt/invoice-4g --artifact-digest sha256:9f2c...

llama-server --model invoice-4g.gguf --host 127.0.0.1 --port 8080 \
  --parallel 8 --ctx-size 4096

mt operator run \
  --serve mt/invoice-4g=sha256:9f2c...@http://127.0.0.1:8080 \
  --concurrency 8
```

You then own keeping up. When a new system wins that arena your digest is stale
and you lose eligibility until you declare the new one. A `--serves-file` json
does the same for several models with per model concurrency and replicas:

```json
{
  "models": [
    {
      "model": "mt/invoice-4g",
      "artifact_digest": "sha256:9f2c...",
      "engines": ["http://127.0.0.1:8080", "http://127.0.0.1:8090"],
      "concurrency": 16
    }
  ]
}
```

Several addresses for one model are replicas. Requests go to whichever is least
busy.

---

## 7 · The wire

Protocol version 2, JSON on one websocket. At most 64 models per connection.

**hello**, the first frame, declaring every model this connection serves:

```json
{"type": "hello", "version": 2, "hotkey": "5G...", "worker": "rack-1",
 "concurrency": 16,
 "models": [{"model": "mt/invoice-4g", "artifact_digest": "sha256:9f2c...",
             "concurrency": 8}]}
```

The gateway checks each model separately. One it cannot confirm is dropped from
the connection and the rest are kept. Only a hello where nothing is admitted is
refused.

**request**, naming the model:

```json
{"type": "request", "correlation": "cmpl_a1b2", "model": "mt/invoice-4g",
 "request": {"prompt": "...", "max_tokens": 512, "temperature": 0.0}}
```

**chunk**, zero or more, as tokens appear. Streaming is what keeps time to first
token real; without it every response looks as slow as its total latency and the
time to first token gate fails everybody.

**response**, one per request, carrying the token ids verification needs:

```json
{"type": "response", "correlation": "cmpl_a1b2", "model": "mt/invoice-4g",
 "text": "...", "finish_reason": "stop",
 "tokens": {"prompt": [1,2,3], "completion": [4,5]},
 "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
```

`finish_reason` is `length` when the token budget ended the generation rather
than the model. A client that trusts the field treats `stop` as a completed
turn, so reporting `stop` on a cut off answer is worse than cutting it off.

**heartbeat** every 15 seconds, **cancel** from the gateway when the client hung
up. Reconnect is exponential to a 60 second ceiling.

---

## 8 · How you are verified

You send tokens. You build no proof and compute no commitment. The validator
does the work afterwards, off your critical path.

It takes your prompt and your token ids and evaluates the whole sequence in one
prefill through its own copy of the certified artifact, keeping logits at every
position. Under greedy decoding it records the margin each returned token lost
by, because honest hardware can disagree only where two candidates were
effectively tied. Under sampled decoding it records the mean negative log
likelihood at the temperature requested.

| verdict | when | costs you |
|---|---|---|
| `pass` | at or under the calibrated threshold | nothing |
| `cheat` | above it, on a calibration shown to separate | cheat share, and collateral |
| `unproven` | anything else | nothing |

`unproven` covers an empty response, a missing calibration, one never shown to
separate a substitution, a prefill that could not be aligned, and greedy and
sampled statistics that disagree. It is counted and never gates.

Admission is a sequential test, not a fixed count. A clean miner is usually
admitted in well under a hundred probes. One validator rejecting keeps you out.

```bash
mt operator status
```

Shows your state, every model you declared, whether each is eligible, and every
probe each validator has run against you.

---

## 9 · What you earn

Per miner and model, over one serving epoch. Only requests somebody paid for
count; free, internal and probe traffic can never earn.

```
score = billed_tokens * reference_rate[model] / 1000
```

The reference rate is set by the network per model, not by what the client paid,
so two miners doing identical work score identically even if one served a
discounted account.

Four gates. Each fails only when the evidence is statistically clear: the lower
bound of a Wilson interval on the failure rate must sit under the threshold, so
a miner with few requests passes unless the evidence is unambiguous.

| gate | threshold | measured on |
|---|---|---|
| success rate | 5 % | every billed request |
| time to first token | 5 % over the model's ceiling | requests that completed |
| time per output token | 5 % over the model's ceiling | requests that completed |
| cheat share | 1 % | verdicts rendered against you |

Ceilings come from the **serving envelope**, published per model. They are not
the arena's class ceilings. The arena measures cost on one CPU thread for
ranking; serving is judged on what a client will wait for. Using the arena
figure here would make the gate meaningless, because it allows two minutes a
query.

A failed gate forfeits that epoch and excludes nobody. You start the next one
clean. Scoring never bans.

Score becomes emission from the serving pool, which is sized by what serving
actually earned and capped as a fraction of what is available. Buying your own
traffic loses money, because the rebate is held below one minus your revenue
share and that bound is enforced in code.

---

## 10 · Troubleshooting

| symptom | cause |
|---|---|
| `an inference miner needs a GPU` | no card found; the agent looked for nvidia-smi, rocm-smi and apple silicon |
| `no engine answering for <model> at <url>` | the engine did not come up; run it by hand and read its output |
| `nothing serves 'onnx'` | that system is in a format no engine here handles |
| `runs at N ms a token here and its envelope allows M` | this machine is too slow for that system; a GPU or a smaller class |
| `404: could not read the catalogue` | pointing at a server that does not serve it; check `--server` |
| `no validator has admitted this hotkey for that model` | you declared a model you are not admitted for; that one is dropped, the rest keep serving |
| `the declared artifact is not the one admitted for that model` | your digest is stale; the arena produced a new winner |
| The plan holds nothing | every model was skipped; read `reasons` in `mt operator plan` |
| Every verdict is `unproven` | that model has no calibration yet, so nobody can be admitted on it |
| Dropped seconds after connecting | liveness, not trust; reconnect and check the machine's outbound path |
| `at declared concurrency` in the logs | the gateway routed past your slots; harmless, it routes elsewhere |
| Admitted but no traffic | nobody is buying that model yet; check `wanted` in the catalogue |

---

## 11 · What is live

| | state |
|---|---|
| Register, collateral, status | live |
| Choosing what to serve from the catalogue | live |
| Fetching, starting engines, declaring, dialling | live |
| Verification by canonical prefill | live |
| Validator probes and admission | live, but blocked on calibration |
| Settlement, gates and the emission split | built, not yet run per epoch |

**Admission waits on a calibrated model.** A threshold is calibrated per model
from honest serving, and used only once the same model at a lower precision
lands on the far side of it. Until then every verdict is `unproven`, which
counts against nobody and admits nobody.

**Nothing is paid yet.** The launch is meter only on free allowance, so nothing
is billed, the serving pool is empty and modelling takes everything. That is the
designed behaviour with no billed traffic, not a placeholder.

---

## Where the code is

| module | holds |
|---|---|
| `microtensor/serving/plan.py` | which systems to hold, given the catalogue and your capacity |
| `microtensor/serving/supervise.py` | fetch, start, declare, redial |
| `microtensor/serving/agent.py` | protocol frames, the engine binding and the run loop |
| `microtensor/serving/client.py` | register, collateral, declare, status |
| `microtensor/serving/canonical.py` | opening an artifact for prefill |
| `microtensor/serving/verify.py` | margins, likelihood, calibration, verdict |
| `microtensor/serving/settle.py` | the four gates and per epoch scoring |
| `microtensor/serving/pools.py` | the pool split and the weight vector |
