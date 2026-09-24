# Serving on Microtensor

You run a certified artifact on your own hardware and answer paid requests with
it. Validators check a sample of what you returned against their own copy of
the same artifact. You earn from the serving pool for tokens somebody paid for.

**Your machine serves; validators verify.** There is nothing to prove while you
generate. You return the tokens you already produced, and the checking happens
afterwards, on the validator, out of your critical path.

This guide is for **inference operators**. You serve models other people built
and you are paid for billed tokens, not per round.

- To build the models instead, read [miner_setup.md](miner_setup.md). That is a
  different role with a different hotkey, measured on reference hardware and
  paid from the modelling pool.
- To rent out hardware by the hour, that is the compute pool, a third module
  again with its own hotkey and its own emission.

---

## 1 · Three roles, one network

| | miner | inference operator | rig owner |
|---|---|---|---|
| produces | an artifact | a service | capacity |
| measured on | reference hardware, by validators | live traffic, by validators | availability, by the compute validator |
| paid for | position on the frontier | billed tokens served | verified uptime |
| settles every | round | serving epoch | compute epoch |
| capital at risk | registration | posted collateral | registration |
| pool | modelling | serving | compute |

They are separate hotkeys. One machine may hold all three and none of them
knows about the others.

---

## 2 · What you run

Two processes.

**Stock inference engines**, one per model you serve. You start them yourself
against the certified artifacts, exactly as published. Nothing is patched,
nothing is instrumented and no custom build is required. Each artifact is pinned
by its manifest digest, and what you serve must hash to that digest.

**One agent.** It dials the gateway over a websocket and holds the connection
open for every model at once. Requests arrive on that socket naming the model
they are for, the agent runs them against the right local engine, and the answer
goes back on the same socket.

Dialling out is the design, not a workaround. You need no inbound port, no
public address, no certificate and no reverse proxy. A machine behind NAT or on
a domestic connection serves exactly as well as one in a rack.

### Serve a pool, and do not pick it

Certified artifacts are small on purpose, between one and sixteen gibibytes,
because the whole point is frontier quality on ordinary hardware. One machine
holds several resident and many more on disk. A process per model would waste
that: ten models would mean ten agents, ten connections and ten copies of
everything.

So one agent serves a set, and the set is chosen for you. The network publishes
which artifacts are certified and how wanted each one is, and a planner on your
machine ranks them against your capacity, fetches what it chose, starts an
engine per model and declares them.

You never type a model name. You would not want to: the catalogue changes every
round as new artifacts win their arenas, and only the network can see which
models have requests and nobody serving them.

Concurrency is the only number you declare about your hardware. You do not
declare a device, a memory figure or a region, because none of those are
verifiable and none of them are what you are paid on. The request shape limits
come from the arena class the artifact was certified under and apply to every
operator serving that model.

---

## 3 · What you return

The completion text, the prompt token ids and the completion token ids. That is
the whole response body, plus a finish reason and the usage counts derived from
those two lists.

```json
{
  "type": "response",
  "correlation": "...",
  "text": "...",
  "finish_reason": "stop",
  "tokens": { "prompt": [...], "completion": [...] },
  "usage": { "prompt_tokens": 128, "completion_tokens": 64 }
}
```

The token ids are what makes verification possible. Without them the validator
would have to re-tokenise your text and guess at the boundaries.

---

## 4 · How you are verified

The validator takes the prompt and the tokens you returned and evaluates the
whole sequence in a single prefill through its own copy of the certified
artifact, with logits kept for every position. It never asks your engine for
anything beyond the tokens.

This is teacher forcing. At each position the validator conditions on **your**
prefix rather than on its own, so one response yields as many near independent
tests as it has tokens, instead of a single divergence event that tells you
nothing after the first disagreement.

### Greedy decoding

Each returned token should be the model's top choice at its position. What is
recorded is the margin it lost by:

```
d_t = max_v logits_t[v] - logits_t[y_t]
```

Honest hardware can disagree only where two candidates were effectively tied,
and a tie carries a margin near zero. A cheaper quantisation agrees on the easy
majority of positions and errs where the model was actually decided, so its
disagreements carry real margin. Both the rate of disagreement and the margin
carried are computed; calibration decides which separates better for that model.

### Sampled decoding

Margins mean nothing when the caller asked for temperature, so the statistic is
the mean negative log likelihood of your tokens at the temperature that was
requested. A substitute model puts systematically less mass on the sequence the
certified artifact would have produced.

### The threshold

It is not a constant and it is not guessed. It is calibrated per model from
honest serving across the hardware generations operators actually run, and it
is only accepted if the same model served at a lower precision scores on the
far side of it. A calibration that was never shown to separate yields
`unproven`, never `cheat`.

### Verdicts

| verdict | meaning | cost to you |
|---|---|---|
| `pass` | the tokens are what the artifact produces | nothing |
| `cheat` | they are not, past the calibrated threshold | counts against your cheat share, and collateral is at risk |
| `unproven` | the sample could not be evaluated | nothing |

An empty response, a missing calibration, a prefill that could not be aligned,
and greedy and sampled statistics that disagree all resolve to `unproven`. An
unusable sample is never held against you.

### What this does and does not establish

It establishes that the tokens you returned are the ones the certified artifact
produces, within a calibrated tolerance. It does not establish which
computation produced them. An operator that returns identical tokens on every
audited request is indistinguishable from an honest one, and the client
received identical output.

---

## 5 · Joining

**Register** under your own hotkey. This earns nothing and serves nothing; it
creates the identity the rest hangs from.

**Post collateral**, a transfer verified on chain before it counts. Collateral
is what makes substitution irrational: it is forfeited on a cheat verdict, and
the deterrence condition is published so you can check the arithmetic yourself.

**Declare what you serve**: a model name and the manifest digest of the
artifact behind it. Declaring a different digest for a model ends your
eligibility for it immediately, because a different artifact is a different
thing to verify against, and you must prove the new one.

**Pass admission.** You serve synthetic probe traffic that no client sees and
that can never earn weight. Each validator runs a sequential test, accumulating
evidence and stopping as soon as the answer is clear rather than after a fixed
count, so a clean operator is usually admitted in well under a hundred probes.
One validator rejecting is enough to keep you out.

Only then does the gateway route a paid request to you.

---

## 6 · Staying admitted

Admission is a state, not a badge. Validators keep verifying a sample of live
traffic, and the eligible set they publish is what the gateway routes on.

Two parties can take you out of rotation, and the difference matters.

- **A validator** withdraws you on trust. Your generation counter advances, the
  probe history from before it stops counting, and you must pass admission
  again before you serve anything.
- **The gateway** may drop you within seconds for liveness: a dropped socket, a
  timeout, a missed heartbeat. That removes you from routing and touches
  nothing a validator decided. Reconnect and you are routable again.

The gateway can never admit an operator and can never withdraw one for any
reason of trust. That separation is why a fast reflex on liveness cannot become
a slow decision about honesty.

---

## 7 · What you earn

Settlement is per operator and model over one serving epoch, and only billed
requests count. Free, internal and probe traffic cannot earn, which is what
stops an operator manufacturing its own volume.

Your score is the billed tokens you served valued at a reference rate the
network sets per model, not at the price the client paid. Two operators doing
identical work score identically even if one served a discounted account.

Four gates apply, and each fails only when the evidence is statistically clear
rather than at a raw ratio. The lower bound of a Wilson interval on the failure
rate must sit under the threshold:

| gate | fails when |
|---|---|
| `success_rate` | too many requests did not complete |
| `ttft_p99` | 99th percentile time to first token is over the ceiling |
| `tpot_p99` | 99th percentile time per output token is over the ceiling |
| `cheat_share` | too large a share of verdicts went against you |

An operator with few requests passes unless the evidence of failure is
unambiguous. A failed gate forfeits that epoch and excludes nobody; the next
epoch starts clean.

Score converts to emission from the serving pool:

```
available = (1 - reserve) * emission
serving   = min(rebate * total_score, cap * available)
modelling = available - serving
```

Two properties hold by construction. With no billed traffic the serving pool is
empty and modelling receives everything. Serving can never grow large enough to
starve the modelling that produces the models it sells, because it is capped as
a fraction of what is available.

Buying your own traffic loses money: the rebate is held below one minus the
revenue share, and that bound is enforced in `Dials.__post_init__` rather than
asserted in a document.

---

## 8 · Where the code is

| module | holds |
|---|---|
| `microtensor/serving/canonical.py` | opening the artifact for prefill and aligning the logit rows |
| `microtensor/serving/verify.py` | margins, likelihood, calibration and the verdict |
| `microtensor/serving/settle.py` | the four gates, the Wilson bound and per epoch scoring |
| `microtensor/serving/pools.py` | the two pool split and the weight vector |
| `microtensor/serving/agent.py` | your side: protocol frames, the engine binding and the run loop |
| `microtensor/serving/plan.py` | which models to hold, given the catalogue and your capacity |
| `microtensor/serving/supervise.py` | acting on the plan: fetch, start, declare, redial |
| `microtensor/serving/client.py` | register, collateral, declare and status, signed by your hotkey |
| `microtensor/serving/probe.py` | the validator side of one probe |
| `microtensor/serving/loop.py` | the validator's probe cycle and what it reports |

---

## 9 · Running it

```bash
pip install -e ".[serving]"
```

### Register and post collateral

```bash
mt operator register --label "my node"
mt operator collateral --reference 0x<extrinsic hash> --block <block>
```

`register` prints how much collateral this network asks for and the coldkey to
send it to. `collateral` is checked on chain before it counts, so report the
transfer only after it is in a block.

### See what the network would give you

You do not pick models. The network publishes which artifacts are certified and
how wanted each one is, and a planner on your machine ranks them against your
disk, memory and slots.

```bash
mt operator plan --slots 8
```

It prints what it would serve, what it would skip and why. Nothing is fetched
and nothing is started. Narrow it with a policy if you want to:

```bash
mt operator plan --track invoice --class mt-4g
```

### Run it

```bash
mt operator run --auto --slots 16 --artifacts /data/artifacts
```

That is the whole thing. It fetches the artifacts it chose, verifies each digest
against the certificate before serving it, starts an engine per model, declares
each one, dials the gateway and answers. It rereads the catalogue periodically
and redials when what it should hold changes.

| flag | means |
|---|---|
| `--slots` | requests at once across every model, the one number you own |
| `--disk-gb` / `--memory-gb` | what you allow; zero means everything free |
| `--max-models` | a ceiling on how many to hold; zero leaves it to capacity |
| `--track` / `--class` / `--not` | narrow what it will consider, repeatable |
| `--engine` | the engine binary to start, `llama-server` by default |
| `--worker` | a name for this machine when several share one hotkey |

### How it chooses

Models rank by how wanted they are, which is requests in the last hour divided
by operators already serving them. A model somebody asked for and nobody serves
rises to the top, so the tail gets covered instead of everyone piling onto the
same popular model. Ties break toward the cold model, then the smaller one.

What you already hold is favoured, so the set does not thrash when two models
are close. Your slots are shared across what it chose, and each model keeps its
own share, so a busy model cannot starve a quiet one.

### Choosing yourself instead

Declare each model and name it:

```bash
mt operator declare --model mt/invoice-4g --artifact-digest sha256:<digest>

llama-server --model mt-invoice-4g.gguf --host 127.0.0.1 --port 8080 \
  --parallel 8 --ctx-size 4096

mt operator run \
  --serve mt/invoice-4g=sha256:<digest>@http://127.0.0.1:8080 \
  --concurrency 8
```

You then own keeping up with the catalogue. When a new artifact wins that arena
your digest goes stale and you stop being eligible until you declare the new one.
A `--serves-file` json pool does the same for several models, with per model
concurrency and replica addresses.

---

## 10 · What is live, and what is not

| | state |
|---|---|
| Register, collateral, status | live |
| Choosing what to serve from the catalogue | live |
| Dialling in and answering routed traffic | live |
| Validator probes and admission | live, but see below |
| Verification by canonical prefill | live |
| Settlement, gates and the emission split | built, not yet run per epoch |

**Admission needs a calibrated model.** A threshold is calibrated per model from
honest serving and is only used once the same model at a lower precision lands
on the far side of it. Until a model carries that calibration, every verdict is
`unproven`, which counts against nobody and admits nobody. Calibration runs are
in progress; watch the repository for the first calibrated model.

**Nothing is paid yet.** The gates, the Wilson bound, the per epoch scoring and
the two pool split are written and tested, but no serving epoch settles on the
coordinator yet, so the serving pool is still empty and modelling receives
everything. That is the designed behaviour with no billed traffic, not a
placeholder.
