# The Microtensor Mechanism

This document is the mechanism specification. It defines what miners optimise
against and what every validator enforces.

Sections marked **frozen** are a commitment: an artifact built against them keeps
competing. Sections marked **tunable** may change with notice and never
invalidate an artifact already submitted.

The governing principle, from which everything else follows:

> **The network holds the weights and runs them.** A miner supplies weights, a
> description of how to load them, and claims about what running them will
> cost; the validator supplies the runtime and produces every measurement. The
> artifact that earns a ranking is byte for byte the artifact a customer
> receives.

Measurement is distributed. A coordinator operated by the subnet owner assigns
each system to at least three worker validators, collects their signed reports,
and publishes a canonical settlement computed from them. The coordinator holds
no protocol privilege: Yuma Consensus takes the stake-weighted median of every
validator's vector and a subnet cannot disable that, so its authority comes
from dominant stake plus a settlement that every worker recomputes from the
published reports before submitting. A worker that cannot reach it settles
standalone rather than halting. See
[coordinator_setup.md](coordinator_setup.md).

---

## 1 · Overview

Miners submit a **compressed model** made of weights, a load manifest, and a
declared envelope, pinned by hash and registered to exactly one track and hardware
class. Every validator, every round:

1. freezes the participant set by artifact digest,
2. derives an identical task set from the chain,
3. **measures** each artifact's deployment envelope on reference hardware,
4. **scores** the admissible ones for accuracy under enforced determinism,
5. sets one weight vector, which Yuma Consensus reconciles by stake-weighted
   median.

The design constraint throughout: **a miner's score must never depend on which
validator evaluated it.** Every deviation from that is called out below and
bounded explicitly.

### Measurement is not scoring

These are separate subsystems, separately tested, and the distinction is load
bearing:

| | Question | Determinism | Module |
|---|---|---|---|
| **Envelope** | Does it fit the hardware? | Device-bound; aggregated by median across conforming validators | `microtensor.envelope` |
| **Accuracy** | How good is it? | Bit-identical between conforming validators | `microtensor.scoring` |

Collapsing them is the most common way subnets end up with hardware-dependent
scores. They are not collapsed here.

---

## 2 · Tracks and hardware classes (frozen per epoch, extensible)

A **track** is a task family. A **class** is a hardware envelope. A submission
binds to exactly one `(track, class)` pair, and competes only within it.

### Tracks

| Track | Enabled | Metric | Emission share |
|---|---|---|---|
| `code` | ✅ | execution pass rate against hidden tests | 1.00 |
| `document` | ⏸ | extraction F1 · span accuracy | none |
| `analytics` | ⏸ | exact-match · numeric tolerance | none |
| `support` | ⏸ | rubric F1 · tool-call correctness | none |
| `detect` | ⏸ | mAP @ fixed IoU | none |
| `vqa` | ⏸ | answer accuracy · grounding IoU | none |
| `speech` | ⏸ | word / character error rate | none |
| `video` | ⏸ | temporal localisation · event F1 | none |
| `image-synth` | ⏸ | reference-anchored perceptual distance | none |

Disabled tracks are registered but carry zero emission share and are not drawn.
Enabling one is a mechanism version bump, not a code change.

The launch scope is deliberately narrow: `code` is the only enabled track and
`mt-3g` its only enabled class, so `CLASS_WEIGHTS` gives that one competition
the whole emission share. One competition means every entrant is measured
against every other entrant under one ceiling, which is the sharpest signal the
mechanism can produce and the fairest starting point for a network with no
history yet. A track may gate the classes it competes in; a code model under the
600 MB `mt-1g` ceiling would sit below the track threshold forever, and a dead
competition still costs every validator fetch and profile time.

Opening a second class is a governance decision, not a code change. It needs a
published `device_profile` and tolerance band for that class first, since
without them envelope conformance cannot be judged and the ceiling would not
actually bind.

**Every enabled metric is computed, not judged.** No model renders an opinion on
another model's output. This is the admission criterion for a track: if quality
cannot be reduced to a deterministic computation against ground truth, it does
not become a track. `image-synth` is the acknowledged boundary case and stays
disabled until its reference-extractor exemption is written down.

### Hardware classes

| Class | Max size `Sₖ` | Max sustained RSS `Rₖ` | Max p95 `Lₖ` | Reference device |
|---|---|---|---|---|
| `mt-16g` | 8 GB | 16 GB | 400 ms | x86-64 server, no accelerator |
| `mt-4g` | 2.5 GB | 4 GB | 120 ms | consumer / embedded GPU |
| `mt-3g` | 1.5 GB | 3 GB | 180 ms | developer workstation |
| `mt-1g` | 600 MB | 1 GB | 300 ms | mobile SoC / NPU |

A class names a memory envelope, not a device. The reference column is the kind
of machine that envelope is meant to fit on, but nothing about it is enforced:
what binds is the ceiling triple, measured on certified validator hardware.

Each class is a standing competition that opens and stays open, so a new class
is an addition rather than a replacement. Anti-stagnation comes from the
rotating corpus partition, incumbent decay, and reference-model advance, each of
which keeps the target moving while preserving work a participant has already
done for a given ceiling.

---

## 2b · Releases

Three layers, each with one job.

| Layer | Cadence | Purpose |
|---|---|---|
| Round | about 24 h | Settles emissions. Miners iterate. Nothing announced. |
| Release | 30 rounds | Freezes the frontier for distribution. The product. |
| Milestone | Per cycle | The stated target for the cycle. The goal. |

A release is a snapshot, not a scoring event. Every 30 rounds the coordinator
freezes the frontier as it settled and publishes it under a version like
`microtensor-code-mt3g-v1`, which names the track, the envelope and the
generation without a lookup. That version is what the API serves, what registry
subscribers download, and what gets announced.

The cutoff is the close block of the final round in the cycle, reusing the
round's own deadline rather than inventing a second one. A system on the
frontier when that round settles is in the release.

Release position is derived from round index the way round index is derived from
block height. Nothing is stored and nobody has to be asked.

**Immutable once published.** A customer who pinned v1 keeps receiving the v1
they pinned, so a correction is a new release rather than an edit. The server
refuses a second release for the same cycle and competition.

**Why separate layers.** No enterprise deploys something that changes daily. A
customer needs a version, a changelog, and a decision about when to upgrade.
Releases give them that while the frontier keeps moving underneath. Serving live
frontier state to a paying subscriber would change their deployment without
their knowledge, which is the failure this layer exists to prevent.

### A release carries no emission weight

This is the decision that is easy to get wrong, so the reasoning is written down
rather than left to be re-derived.

The obvious addition is a bonus for making the release frontier. It must not be
added. A release bonus makes the cutoff round worth more than the twenty-nine
before it, which rewards holding an improvement back until just before cutoff.
That is sandbagging, and it is exactly what continuous daily rounds exist to
prevent. It would also concentrate all measurement load into a single round.

Emissions stay per round, settled per round, submitted per epoch. The release is
reputational and commercial: the system ships to customers under a version with
its miner's name on it, and that is the reward.

If this is ever revisited, the safe form is a small standing bonus for
consecutive rounds on the frontier, which rewards sustained quality rather than
timing.

### Milestones

A milestone is a target quality and cost stated at the start of a cycle and
carried in the release manifest with whether it was met. It is descriptive.
Nothing gates on it and nothing pays for it, and it is deliberately held outside
the anchored config so changing it is not a consensus event.

If a system meets it, that is the result for the cycle and the next target is set
harder. If nothing meets it, the release ships with the best available and the
target rolls forward.

---

## 2c · Telemetry

A miner may report what it is doing: which phase it is in, which epoch it has
reached, its loss, its throughput, and what hardware it claims to be training
on. The coordinator stores that stream and serves it.

**It is observational and nothing more.** Telemetry is reported by the party
being measured, so it can be neither verified nor trusted. It therefore never
enters a certificate, never affects a score, never gates admission, and never
determines eligibility for a competition.

The whole design follows from that one boundary: a miner can tell you what it is
doing without ever telling you how good it is. What it is doing is reported;
how good it is, validators measure on their own certified hardware.

Hardware is the clearest case. A participant can claim any accelerator, which is
acceptable for describing the network and unacceptable as an input to anything
else. Peak throughput is derived from the accelerator name through a table in
this repository rather than accepted from the miner, so the figure means the
same thing across the network and cannot be inflated.

Liveness is the one place telemetry touches the mechanism, and it is bounded.
Silence past `TELEMETRY_HEARTBEAT_BLOCKS` drops a miner from the round, measured
in blocks rather than epochs because a silent miner has no epoch clock and
blocks are chain derived. A chain commitment ends the check: the artifact is
already fetchable, and discarding it over a dead heartbeat would throw away real
work for a reason unrelated to the work. The dropped set is published in the
settlement so a worker verifies it rather than taking the coordinator's word,
and a standalone validator, which sees no telemetry at all, enrols from
commitments and settles identically.

---

## 3 · The submission (frozen)

Three parts, signed by the miner's hotkey and pinned by digest:

**1 · Weights**, in a portable format the validator has an engine for:
`onnx` or `gguf`. `safetensors` is defined in the protocol and not yet
executable — see the engine list below.

**2 · A load manifest**, declaring format, quantisation, preprocessing, the
maximum input the artifact supports (context length, resolution, duration, frame
count as the track requires), and the entry points by which the validator loads
and invokes it.

> The manifest is a **declaration, not a program**. The validator supplies the
> execution engine. **No miner-authored code ever runs on validator hardware.**

This is strictly safer than executing submitted code, and it has a cost that is
stated rather than hidden: architectures are bounded to what the engine
implements. The engine contract is pinned in §9.

**3 · A declared envelope** `ê(a) = (ŝ, m̂, ℓ̂)`, giving the size, sustained peak
memory, and p95 latency the miner asserts the artifact will exhibit on the class
reference device.

### Size is total

```
size(a) = Σ bytes(f)  for every f required to execute the model
```

Weights, adapters, tokenizer, preprocessor, configuration, class maps, auxiliary
tensors. Measuring only the principal weights file would let a miner relocate
capacity into a companion adapter and clear the ceiling while shipping something
larger.

### Caps

| Bound | Value | Enforced by |
|---|---|---|
| Artifact size | class ceiling `Sₖ` | admissibility gate |
| Manifest | 64 KiB | submission API |
| Submissions per hotkey | 1 per 6 hours | submission API |
| Slots per `(track, class)` | 40 | submission API |

A slot frees when its hotkey deregisters, or is evicted after **six consecutive
rounds** below the track threshold. Parking a dead artifact does not hold a slot.

---

## 4 · Rounds (tunable: length)

A round is a window of blocks. Boundaries are computed from block height, so
every validator agrees on when a round opens with no coordination and no
registry to disagree with.

```
index(b)   = (b − GENESIS) // ROUND_BLOCKS
start(e)   = GENESIS + e · ROUND_BLOCKS
close(e)   = start(e) + ROUND_BLOCKS − SUBMISSION_CLOSES_BEFORE_BLOCKS
end(e)     = start(e) + ROUND_BLOCKS − 1
deadline(e)= end(e) − DEADLINE_MARGIN_BLOCKS
seed_e     = H(hash(B_close) ‖ track ‖ class)
```

The round has three phases. `[start, close)` accepts submissions. `[close,
deadline]` is the evaluation window of 560 blocks, about two hours. The last
`DEADLINE_MARGIN_BLOCKS` are slack for the weight extrinsic to be included
before the round turns over.

**The seed is drawn from the close block, not the start block.** Seeding from
the start block would let anyone who reads the chain compute `seed_e` and then
submit inside the same round with the task set already in hand. Because
submissions close at exactly the block whose hash becomes the seed, no submitter
can have seen it. The hash is still public immediately afterwards, so selection
stays independently reproducible.

At the close block the participant set **freezes** by artifact digest. A version
committed after the freeze competes in the next round.

### The pointer on chain is small on purpose

A miner does not upload a model to the chain; it commits a fixed-width pointer
and serves the artifact itself.

```
mt1|<round>|<track>|<class>|<digest-128>|<scheme>:<locator>
```

Everything is length-bounded so the whole record fits the 128-byte commitment
limit with room for the locator. The digest is the leading 128 bits of the
manifest's SHA-256; the manifest carries the full digest of every file, and the
validator re-derives and re-checks it after the fetch. Truncation here costs
nothing, because the short digest only has to bind the pointer to one manifest, and a
128-bit second-preimage is not a budget anyone has. Anything the chain cannot
hold, meaning the load manifest, the declared envelope and the signature, lives in the
manifest, addressed by that digest.

Unparseable commitments are skipped, not rejected loudly. A malformed pointer
costs its author the round and costs the validator one `continue`.

### The corpus is partitioned, and the two halves do different jobs

```
T_e = T_rot(e) ∪ T_fix          T_rot ∩ T_fix = ∅
|T_rot| = ⌈0.7 N⌉   drawn with seed_e
|T_fix| = ⌊0.3 N⌋   constant across rounds
```

**The rotating partition defeats overfitting.** A miner cannot tune against
tasks it cannot predict.

**The fixed partition preserves comparability**, and it is not redundant.
Without a constant slice, a leaderboard movement is ambiguous between "the models
improved" and "the generator drew an easier batch." The fixed partition is the
invariant that turns a movement into a claim about capability, and it is the
series published externally as evidence the network works.

Both are drawn from distributions absent from public corpora. The fixed
partition rotates only on detected contamination, and any rotation establishes a
replacement baseline in parallel so the historical series stays interpretable.

### Prompt wording is load bearing

A track prompt is frozen into the corpus and served to every artifact, so a
wording defect becomes a property of the whole arena. The extract track found
the canonical trap. Its prompt originally ended with the bare escape clause
"Return an empty list if there are none." The 0.6B baseline obeyed the clause on
nearly every sentence and scored micro-F1 0.057 at precision 1.000 with zero
format failures. Every output was a valid empty entity list, so nothing in the
harness flagged it. Rephrasing the clause to name the condition and the exact
shape to return recovered the baseline to roughly 0.37.

Two rules follow. Never end a prompt with a bare escape clause; state the empty
case with its condition and its exact output shape. And treat a near-zero
baseline with perfect precision and clean formatting as the signature of a model
taking an escape hatch, not as evidence of task difficulty.

---

## 5 · Envelope measurement (frozen)

The gate means nothing unless the numbers behind it reflect service conditions.

### Memory is measured at load, under load

Resident memory decomposes into a constant part and an input-dependent part:

```
m(x) = m_w + m_a(x)
        │      └── activation and cache: zero at load, not zero in service
        └── weights
```

`m_a` grows differently per architecture family:

| Family | Growth | Consequence |
|---|---|---|
| autoregressive | `2·L·h·d·c·b`, linear in context | passes at short context, exceeds at long |
| vision / conv | `≈ β·φ·HW/s²·b`, quadratic in linear resolution | 640² → 1280² is ~4× |
| iterative generative | ≈ constant across steps | memory and latency have different governing parameters |

So the measurement is taken at the artifact's **declared maximum input** under
sustained batch, sampling RSS throughout and recording the maximum:

```
m̂(a) = max over t ∈ [0, T_prof] of RSS(a, c_max, β_max, t)
```

Not the mean, not the steady state, not the value at load. `input_at_peak` is
recorded in the certificate in the track's own terms, so a deployer can tell
whether their workload is covered by the measurement.

### Latency is a distribution

Cold start is measured from process launch to first emitted output on an
unwarmed cache. Steady state is measured over a sustained request stream:

```
ℓ_p95(a) = quantile_0.95 { ttft_i }      gated against the class ceiling
c_front(a) = quantile_0.95 { total_i }    the cost of a front-only system
```

A request that produces no output is a failure, not a sample. A profile in
which no request produces output faults the artifact instead of measuring
it, so a system cannot record a cost of zero by answering nothing.

A single warm measurement is not admissible evidence, because it is precisely
the measurement a miner would construct if permitted to construct one.

### Budgets are CPU-time, not wall-clock

Enforced through cgroup accounting, with a wall-clock backstop at **3×** purely
to catch runs that sleep or stall on I/O. Faster validator hardware must not
change an outcome.

### Reference hardware

Every envelope quantity is measured on the published reference profile for the
class, identified in the certificate by `device_profile` hash. A validator whose
hardware does not conform **may still score accuracy**, since that is
hardware-independent, but its envelope measurements are excluded from
aggregation. Conforming measurements aggregate by **median**.

---

## 6 · The admissibility gate (frozen)

Binary. An artifact either fits the class or does not exist for the round.

```
g(a) = [ s(a) ≤ Sₖ ] · [ m(a) ≤ Rₖ ] · [ ℓ(a) ≤ Lₖ ]      ← fits the class
     · [ m(a) ≤ m̂ ]  · [ ℓ(a) ≤ ℓ̂ ]                       ← declaration holds
```

Two distinct failures:

- **Exceeds the class ceiling** → inadmissible, because it does not fit the
  hardware.
- **Within the ceiling but exceeds its own declaration** → inadmissible, because
  its certificate would be false. *A false certificate is worse than no
  certificate: it transfers a hidden failure to the deployer who trusted it.*

### Why declaration is truthful in equilibrium

No term of the score depends on `m̂`. The incentive comes entirely from an
asymmetry:

- Declaring **below** the truth fails the gate and scores zero. The loss is
  total, not marginal.
- Declaring **far above** the truth passes, but writes a pessimistic number into
  the certificate. Deployers provision against the declaration and select on
  footprint when accuracy is comparable, so an inflated declaration loses
  commercial selection. The loss is real but bounded.

The optimum is therefore

```
m̂* = m + z·ς
```

for a small confidence multiplier covering measurement variance across
conforming devices. **The miner declares the truth plus a margin it can defend,
which is exactly the number a deployer needs.**

Ceilings describe a class. Declarations describe an artifact. A deployer
provisions against the declaration, which is finer-grained and verified.

---

## 7 · Accuracy (frozen)

For an admissible artifact, per-task correctness `q(a,t) ∈ [0,1]` is computed by
the track metric against ground truth. Accuracy is the mean over the **full**
task set:

```
A(a) = (1/|T_e|) · Σ q(a,t)
```

The sum runs over every task, not only those the artifact handled. A task it
fails, refuses, or times out on contributes zero. **Omission is never free**, so
an artifact cannot inflate its position by attempting only what it finds easy.

### The final score

```
Σ(a) = g(a) · A(a)
```

**This is the central design decision of the subnet.** A composite alternative
exists and is worse:

```
Σ(a) = A(a)/P(a) · 1/ℓ(a) · 1/m̂(a)        ✗
```

It fails four ways. **Unboundedness.** As `ℓ → 0` the score diverges, so scores
are not comparable across rounds, tracks or classes. **Substitutability.** A
product permits arbitrary trade, so a near-useless model with tiny memory can
outrank a strong one; below an accuracy floor a model is worthless at any
footprint. **Gradient misdirection.** Latency and memory admit large cheap
gains through aggressive quantisation and accuracy does not, so the formula
instructs the network to produce very small models that do not work.
**Dimensional incoherence.** Accuracy over perplexity times inverse
milliseconds times inverse bytes is a quantity with no interpretation, which
cannot be audited or explained to a customer.

Gate-then-rank is bounded in `[0,1]`, admits no substitution across the
constraint boundary, directs all optimisation pressure to accuracy once the
envelope is met, and maps exactly onto the question a buyer asks: **does it fit
my hardware, and how good is it.**

### Enforced determinism, stated per family

| Family | Requirement |
|---|---|
| discriminative (`detect`, `speech`, `vqa`, `video`) | single forward pass with argmax/threshold, deterministic by construction once runtime, precision and preprocessing are pinned |
| autoregressive (`code`, `document`, `analytics`, `support`) | greedy decoding, temperature 0, pinned engine version |
| iterative generative (`image-synth`) | fixed seed, fixed scheduler, fixed step count, published per round |

Under these conditions inference is a deterministic function, so for validators
`v, v'` on the same artifact and task set: `A_v(a) = A_v'(a)`.

### The scored object is a system, not a model

A deployer does not buy a model, they buy a composition: a compact front model
answering what it can, a router deciding, and a larger specialist taking the
remainder. Expected cost per query and end-to-end quality are properties of that
composition, so those are what the network measures.

A system is `S = (f, r, g)`. The front is bound to the competition class and
runs on every query. The router is bound to the front, since it is co-resident
on the constrained device, and its bytes and memory count against the front's
envelope. The specialist is bound to the host profile and sees only escalated
queries, so its cost enters weighted by the escalation rate.

**A router is data, never code.** It arrives as a threshold table or a small
ONNX graph over a fixed feature list, and the validator interprets it. Features
are computed by the validator from what the front emitted; nothing an artifact
reports about itself reaches the routing decision. That is what makes the purity
condition checkable rather than a promise, and it is why the new execution
surface opens no hole in the sandbox posture.

**Quality is end-to-end.** The track metric is applied to what the system
finally answered, never to the front in isolation. The front's own score is
recorded for diagnostics and never ranked, because a cascade is bought whole.

**Cost is measured, not modelled.** The profiler still establishes the envelope
that admits a system; execution establishes the cost that ranks it, from the
timings of this round's actual run.

**A system with no router and no specialist is a valid system.** It reduces to
the single-artifact path exactly, which is how the network runs today.

### Emission follows the frontier, not a rank

Frontier standing decides the order; payment runs down a fixed ladder. The top
eight positions receive 30/20/14/11/9/7/5/4 of the miner pool: frontier members
first, by exclusive hypervolume, then the remaining eligible systems by the area
each covers under the reference cost. Two deployed values ride in the anchored
round config and are worth knowing: `reference_cost_ms` is 10000, and
`min_rounds_observed` is 1, so a first-round submission is eligible at once.

Among admissible systems there is no further scalar. Systems are placed on the
cost-quality plane, the eps-non-dominated set is the frontier, and emission is
proportional to exclusive hypervolume: what the frontier loses if that system
were not there.

Both coordinates are quantised to an integer grid before any comparison, so
frontier membership is bitwise identical across validators rather than
approximately agreed. A cheaper system at lower quality is not beaten by an
expensive better one, so discovering a genuinely new trade-off point pays
wherever it occurs, which is what a rank ordering cannot express.

Two systems sitting inside one epsilon-neighbourhood collapse to the earlier
commitment. Epsilon-dominance alone would leave both on the frontier, and under
leave-one-out each would then measure near-zero exclusive area, because neither
covers much the other does not. As a competition matures and the frontier
clusters, that would collapse total exclusive hypervolume and make emission
noisy, letting a mediocre but isolated system out-earn two good ones standing
together. Collapsing is also the right answer on its own terms: the later
submission added nothing the earlier one had not already provided.

Within a system, each component is priced by role-baseline ablation: the
measured change in the system's exclusive hypervolume when that component is
replaced by the published baseline for its role, substituted in place rather
than added alongside. A component the system does not need measures zero and
earns nothing. **A component whose baseline is unpublished reports null, never
zero**, because the network could not measure it, and reporting a measurement it
did not make would be the same error as scoring zero for an infrastructure
fault.

---

### What a training run does and does not establish

Every submission must carry a public run in `microtensor/training-runs`, named
for the miner hotkey and carrying the artifact digest in its summary. A
submission without one is rejected at discovery, in the same class as an
unsigned manifest. This is an admission requirement; among admitted artifacts
ranking is unchanged and remains gate-then-rank on accuracy alone.

**A run is a timestamped record, not a proof of computation.** Entries are
written by the miner through an API, and a determined actor can fabricate
plausible curves. What the requirement establishes is that anyone claiming an
artifact must also have published a training history bound to that artifact's
digest before submitting it, which raises the cost of appropriation from a
download to a fabrication and leaves an auditable public trail.

**It is not what makes the score trustworthy.** That remains the fact that the
network holds the weights and runs them itself, on its own certified hardware,
against tasks seeded from a block hash that did not exist when the artifact was
submitted. Provenance is an audit trail layered on top of that, never a
replacement for it.

**The artifact-derived signals are the ones the miner cannot author.**
`base_divergence` and `uplift` are computed by validators from the weights
themselves. Together the three answer the question completely: a public record
of how the model was produced, and two independent measurements confirming it
genuinely moved from its base and genuinely improved on it.

**A validator that cannot reach the run store abstains.** It does not skip the
check, and it does not score only the miners it managed to verify. Two
validators with different reachability would otherwise materialise different
participant sets and diverge by construction, which is precisely the property
the rest of this document exists to protect.

---

### The metric is pass@1, and that is forced rather than chosen

`pass@k` estimates the probability that at least one of *k* sampled generations
passes the hidden tests. Any `k > 1` requires temperature sampling, and sampling
is non-deterministic: two validators drawing different samples from the same
artifact would compute different scores, and the divergence would be expected
rather than attributable. That destroys the property §7 exists to protect.

So the metric is **pass@1 under greedy decoding**: one generation at temperature
0, executed, pass or fail per hidden test. This is also the honest match to
deployment, where a served model returns one answer rather than ten for a
harness to choose between.

The internal metric id stays `execution_pass_rate`. The certificate publishes
both that id and `pass@1 (greedy)` as `metric_display`, so machine consumers
keep a stable key while the number stays comparable to published results.

### Score quantisation, making that literally true

Floating-point reduction order is not associative. Two conforming validators can
differ in the last decimals from SM count, cuDNN algorithm selection, or batch
shape, **even on a pinned runtime version**. A bare equality claim would be
false in practice.

So the mechanism requires deterministic kernels as part of the device profile
(`torch.use_deterministic_algorithms(True)`, fixed cuBLAS workspace, TF32
disabled), **and then quantises the score to a fixed grid before weight-setting**:

```
A_reported = round(A, ACCURACY_DECIMALS)      # ACCURACY_DECIMALS = 4
```

The grid is far coarser than achievable FP noise. Conforming validators agree
**exactly**, and "divergence is diagnostic rather than expected" becomes a
statement that is true rather than aspirational.

---

## 8 · From scores to weights

Per round, each validator:

1. Applies the gate, then the track threshold `τ`, and ranks the survivors
   within each `(track, class)`.
2. Requires `MIN_ROUNDS_OBSERVED` before an artifact can hold weight at all. A
   new artifact scores, but scores zero weight until there is enough evidence,
   *unknown is not the same as good.*
3. Pays a geometric schedule over `K` paid ranks:
   ```
   α_j = (1-γ)·γ^(j-1) / (1-γ^K)
   ```
4. Applies **hysteresis at every rank boundary**, not only the top.
5. Applies **incumbent decay**.
6. Applies the **concentration cap**.
7. Normalises over all UIDs, then smooths:
   `w = α·w_new + (1-α)·w_prev`, with the prior persisted.
8. Sets the vector on chain, once per round. When the subnet's
   `commit_reveal_weights_enabled` hyperparameter is on, the runtime routes the
   extrinsic through commit-reveal transparently; the validator queries the
   hyperparameter each round and logs which path was taken, so an operator can
   see whether their vector is being revealed late or published immediately.

### Hysteresis at every boundary

```
rank j is taken from the holder only if   Σ(challenger) > Σ(holder) + ε
```

The whitepaper originally specified this for rank 1 only. That is not enough.
The empirical basis is SN9's documented winner-take-all hoarding: a single
dominant miner absorbing effectively the whole emission pool killed the
incentive for anyone else to compete. The geometric schedule with a paid tail
is the direct response.

With `γ = 0.85, K = 8`, rank 2 pays ~15% of the track pool, so copying the leader
and landing second is a **very profitable** attack that a rank-1-only margin
never touches. Applying `ε` pairwise removes the return on appropriation
everywhere, which is a stronger defence than detection because it operates
*before* the copy is made.

### Incumbent decay

```
α̃₁(n) = α₁ · (1-δ)^n
```

where `n` counts consecutive rounds without a **material** improvement, defined
as clearing `ε`, not merely resubmitting. Without that definition an incumbent
resets `n` every round with a re-seeded retrain and the decay becomes a cosmetic
resubmission tax. The attenuated remainder redistributes down the schedule.
Rank is unchanged; only share moves.

### Concentration cap

Miners are grouped by **coldkey**. A miner is reachable only through the
chain, so an IP-prefix
grouping keys on an address that is absent or stale and misses its actual
target, one operator running many hotkeys; the coldkey is what that operator
cannot cheaply multiply without splitting stake. If any coldkey holds more than
`CONCENTRATION_CAP_FRACTION` of the paid cohort, the excess is zeroed lowest-first
and the vector renormalised. Registration is permissionless and cheap; without
this, one operator fielding a slate takes a whole track.

### Asymmetric trust

Weight moves toward a new score faster on the way **down** than up
(`DECAY_RATE > RECOVERY_RATE`). A regression is acted on immediately; a recovery
is earned back. Symmetric smoothing lets an artifact alternate between good and
broken while holding a mid position.

---

## 9 · The engine contract (frozen)

Because no miner code runs, the validator's engine defines what can be
submitted. Stating that precisely is a mechanism obligation, not an
implementation detail.

- **Format**: `onnx` (opset pinned per mechanism version) and `gguf` (file
  version pinned). Both ship as optional extras; `mt inspect engines` reports
  what a given validator actually registered, and a format with no engine
  cannot be submitted against.
- **ONNX**: standard opset only. **Custom operators are rejected at submission.**
- **GGUF**: greedy decoding, one thread, no GPU offload, seeded, against an
  exactly pinned llama.cpp build. The architecture a file declares is checked
  against the set the engine runs before llama.cpp opens it, so an
  unsupported model is rejected by name rather than as a loader crash. Moving
  the pin is a real change — the bundled llama.cpp decides which
  architectures load at all — and CI loads one real model per declared
  architecture on every push to prove the pin still can.
- **safetensors**: defined in the protocol, **no engine**. Executing raw
  weights means a framework, and a framework's kernels are selected at runtime
  by host and build. Two workers would disagree on quality for reasons no
  miner could act on, which is a reason to withhold the format rather than to
  relax the determinism requirement. It stays in the enum because the protocol
  is versioned and removing a value is a breaking change; submissions
  declaring it are refused for want of an engine.
- **Execution providers**: pinned per class in the device profile.
- **Determinism flags**: mandatory, and part of the profile hash.

Determinism across hosts rests on the certified reference device in every
case: both engines dispatch CPU kernels by instruction set, and AVX2 and
AVX-512 do not reduce identically. Quality is only ever compared between
workers measuring on the arena's certified device.

Architecture search is bounded by this. That is a real cost of refusing to run
miner code, and it is stated rather than discovered.

---

## 10 · Failure semantics (frozen)

> **A fault of the artifact scores zero deterministically. A fault of the
> infrastructure abstains.**

Artifact faults are content-determined and identical on every validator.
Infrastructure faults differ per validator and must never leak into a vector.

| Event | Consequence |
|---|---|
| Exceeds class ceiling on any axis | Inadmissible, scores 0 for the round |
| Measured envelope exceeds its own declaration | Inadmissible, scores 0 |
| Missing entrypoint, wrong track/class, digest mismatch | Scores 0 |
| Custom operators, unsupported format | Rejected at submission |
| Manifest declares an input the engine cannot honour | Scores 0 |
| Task times out or produces no output | That task scores 0 |
| Flagged as a copy (§11) | Later submission scores 0 |
| Artifact unfetchable after retries | **Validator abstains** |
| Reference device unavailable / profile mismatch | **Envelope excluded; accuracy may still be scored** |
| Engine unavailable | **Validator abstains** |
| Round window closes with a track incomplete | **Validator abstains** |

Abstaining means setting no weights that round. EMA state is untouched and the
validator resumes next round. **A partial vector is never submitted**: with all
tracks per validator, a missing track removes that track's whole emission share,
which is a *consensus divergence, not a smaller sample*.

Artifacts are materialised and digest-verified for every track **before**
execution begins, so fetch failures surface at round start rather than mid-round.

---

## 11 · Plagiarism (frozen)

Artifacts are published, so a competitor may download a leader, perturb it, and
resubmit. Detection must distinguish that from a case that looks identical and is
entirely legitimate.

**The convergence problem.** Two miners independently distilling from the same
teacher, on the same task distribution, under the same ceiling, produce models
that behave similarly. *That convergence is the mechanism working.* A naive
behavioural cutoff would punish it, and hardest in exactly the tracks where
competition is healthiest.

So detection requires **two independent signals in agreement**:

```
D_θ(a,b) = 1 − ⟨θ_a, θ_b⟩ / (‖θ_a‖‖θ_b‖)          parameter distance
D_π(a,b) = (1/|P|) Σ JS( π_a(·|x) ‖ π_b(·|x) )     behavioural distance

flag ⟺ ( D_θ < θ_θ )  ∧  ( D_π < θ_π )
```

Parameter distance alone false-negatives, because retraining from a copied
initialisation moves weights while preserving behaviour. Behavioural distance
alone false-positives on legitimate convergence. The conjunction is required.
`P` is a secret probe set. Jensen-Shannon over KL for symmetry and boundedness.

On a flag, the **earlier commitment holds position and the later scores zero.**

Detection is the second line of defence. The `ε` margin of §8 is the first and
the stronger, because it removes the economic return before any copy is made.

---

## 12 · Evaluation cost (tunable ordering, frozen budgets)

```
c_acc(a) = |T_e| · w / θ_k          θ_k = sustained throughput on class k
c_env(a) = fixed by the profiling schedule
C_v      = |A| · ( c_acc + c_env )
```

`θ_k` appears in the denominator: throughput rises as the class ceiling falls,
because a smaller artifact holds fewer weights in bandwidth-limited memory.
Therefore

```
S_k ↓  ⇒  θ_k ↑  ⇒  C_v ↓
```

**Evaluation cost falls as the deployment constraint tightens.** The most
commercially valuable classes are the cheapest to verify, so the network can
afford to run its most valuable tracks most often.

### Ordering is per track, chosen by the ratio

Profiling and accuracy scoring differ in cost by track, often by two orders of
magnitude. Running them in a fixed order wastes the cheaper one:

- `c_env ≫ c_acc` → **score accuracy first, profile only the top cohort.**
  Profiling an artifact that would rank 30th is wasted.
- `c_acc ≫ c_env` → **profile first**, since the gate eliminates candidates
  before the expensive scoring runs.

This does not weaken gate-then-rank: the gate remains binary and dispositive.
Only the order of evidence-gathering changes, and any artifact that fails the
gate scores zero regardless of when it was measured.

---

## 13 · The Verified Model Certificate

The commodity the subnet produces. Content-addressed, machine-verifiable, signed
by the executing validator.

```
VMC {
  track, class,
  artifact  { weights_hash, manifest_hash, total_bytes, format, base_model, round },
  envelope  { size_bytes, peak_rss_bytes, input_at_peak,
              ttft_p50_ms, ttft_p95_ms, total_p50_ms, total_p95_ms, tok_per_sec,
              cold_start_ms, device_profile },
  accuracy  { score_fixed, score_rotating, score_combined,
              n_fixed, n_rotating, corpus_version, metric },
  runtime   { decode, temperature, seed, engine_version, quantization },
  attestation { miner_hotkey, validator_hotkey, signature, canonical_hash }
}
```

Verification is offline and requires trusting no one:

```
C(m)  = sort_keys( NFC( serialize(m) ) )      canonicalisation
η     = SHA256( C(m \ attestation) )
verify( pk_validator, η, σ ) = true  ⟺  authentic
```

NFC normalisation removes composed-versus-decomposed ambiguity; key ordering
removes serialisation ambiguity. A verifier additionally checks `weights_hash`
against the round's pinned submission record to confirm which miner's artifact
produced the measurement.

> **No model participates in issuing a certificate.** Every field is a direct
> measurement or a deterministic function of one. A certificate contains no
> generated prose and no model-authored judgement.

---

## 14 · Frozen versus tunable

**Frozen.** Miners build against these, and changing them is a mechanism version bump:

- The track and class tables, and their emission shares
- Chain-seeded shared task selection, and the 70/30 partition
- The scoring function `Σ = g · A`, its metrics, thresholds, and quantisation grid
- The submission contract: manifest schema, declared envelope, digest pinning
- The engine contract of §9
- Envelope measurement procedure and budgets
- The failure semantics table
- The plagiarism criterion

**Tunable with notice.** Never invalidates a submitted artifact:

- Round length, slot caps, eviction window
- `γ`, `K`, `ε`, `δ`, EMA `α`, decay/recovery rates
- Concentration cap fraction
- Per-track evaluation ordering
- Commit-reveal interval
- Validator isolation internals and the registry serving path

---

## 15 · Why this is not the same as scoring behaviour

A subnet that scores a live endpoint or an agent inherits non-determinism,
uptime dependence, and unattributable divergence. Honest validators legitimately
differ, and a dishonest one hides inside that spread.

Microtensor scores a **frozen artifact under pinned decoding**. Two correctly
configured validators compute the identical quantised score. Divergence is
therefore attributable to the validator producing it, whether a non-conforming
engine, a corrupted corpus copy, or dishonesty, rather than excusable as sampling.

Consensus continues to bound the damage. Determinism additionally **identifies
the source.**
