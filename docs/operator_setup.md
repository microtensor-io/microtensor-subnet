# Serving on Microtensor

You run a certified artifact on your own hardware and answer paid requests with
it. Validators check a sample of what you returned against their own copy of
the same artifact. You earn from the serving pool for tokens somebody paid for.

**Your machine serves; validators verify.** There is nothing to prove while you
generate. You return the tokens you already produced, and the checking happens
afterwards, on the validator, out of your critical path.

If you are here to build models, read [miner_setup.md](miner_setup.md). If you
are here to rent out hardware by the hour, that is the compute pool, which is a
different subnet module with its own hotkey and its own emission.

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

**A stock inference engine.** You start it yourself against the certified
artifact, exactly as published. Nothing is patched, nothing is instrumented and
no custom build is required. The artifact is pinned by its manifest digest, and
what you serve must hash to that digest.

**The agent.** It dials the gateway over a websocket and holds the connection
open. Requests arrive on that socket, the agent runs them against your local
engine, and the answer goes back on the same socket.

Dialling out is the design, not a workaround. You need no inbound port, no
public address, no certificate and no reverse proxy. A machine behind NAT or on
a domestic connection serves exactly as well as one in a rack.

You declare one number about your hardware: concurrency, the number of requests
you will take at once. You do not declare a GPU model, a memory figure or a
region, because none of those are verifiable and none of them are what you are
paid on.

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
| `microtensor/serving/agent.py` | the operator side protocol frames and request handling |

---

## 9 · Availability

The verification, settlement, emission and routing logic is built and tested.
The operator client is not yet released: `Engine.generate` raises
`NotImplementedError` until an engine is bound, there is no run loop around it,
and the server exposes no operator routes yet. There is nothing to install and
no command to run today.

This page gains its install and run commands when those land. Watch the
repository for the release.
