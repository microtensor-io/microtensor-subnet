# Microtensor · netuid 70

A Bittensor subnet that pays for **frontier accuracy at one four-hundredth the
size**. Models are compressed into hardware-constrained specialists and held to a
*verified* deployment envelope.

Most inference subnets score behaviour: send a prompt, judge the answer. That
measures a service under conditions the miner partly controls. Microtensor
measures the artifact itself on certified reference hardware, checking size,
resident memory at declared maximum input and tail latency. It gates on those
numbers, then scores accuracy only among the models that fit.

```
measurement  →  gate (binary)  →  accuracy (only among survivors)  →  weights
```

## Quick start

```bash
pip install ".[validator]"     # validators
pip install "."                # miners need nothing else

mt inspect tracks              # competitions, ceilings, emission shares
mt inspect engines             # what this host can execute, sandbox status
```

Everything defaults to **netuid 70** on finney. Pass `--netuid` only for testnet.

- **Mining** → [docs/miner_setup.md](docs/miner_setup.md)
- **Validating** → [docs/validator_setup.md](docs/validator_setup.md)
- **The mechanism, in full** → [docs/mechanism.md](docs/mechanism.md)

The two operator guides share almost nothing, because the two roles share almost
nothing. Miners run no neuron, serve no inference, and ship no container. They
publish an artifact and commit an 84-byte pointer. Validators do everything else.

## How a round works

```
[start ─────────────────── close) [close ──────── deadline] [·· margin ··]
 submissions open           freeze  evaluate + settle          extrinsic slack
```

The round seed is drawn from the hash of the **close** block, the exact block at
which submissions stop. No submitter can have seen the task set, and anyone can
reproduce the selection afterwards.

1. **Discover.** Decode on-chain pointers, fetch each manifest, verify its
   signature and that it hashes to what was committed.
2. **Materialise.** Fetch every artifact and re-derive its full file-tree digest
   before any execution begins.
3. **Profile.** Cold start, then sustained load at declared maximum input,
   sampling RSS throughout. Inside a resource jail.
4. **Gate.** Over a class ceiling, or over its own declaration, and the artifact
   does not exist this round.
5. **Score.** Chain-seeded task set, 70 % rotating and 30 % fixed, quantised to 4
   decimals so validators agree bit-for-bit.
6. **Settle.** Geometric ranks with hysteresis, incumbent decay, concentration
   cap, asymmetric EMA blend, one vector.

## Design commitments

**Measurement is not scoring.** The envelope is a hard gate, not a term in a
weighted sum. A model that misses the memory ceiling scores zero regardless of
how accurate it is.

**Declaring honestly is the dominant strategy.** Over-declare and your published
certificate is weaker than a competitor's. Under-declare and you are
inadmissible. The 2 % tolerance applies to your declaration, never to the class
ceiling.

**Artifact faults score zero; infrastructure faults abstain.** A model that hangs
or crashes is the model's problem and is scored deterministically. A source that
is unreachable is the validator's problem, so it sets no weights that round
rather than publishing a partial vector. A missing track is a consensus
divergence, not a smaller sample.

**Fail closed.** A host that cannot enforce CPU and memory limits refuses to
execute untrusted artifacts. Every jail result carries `sandboxed`, so an
unenforced measurement can never reach a weight vector.

**Determinism is enforced, not hoped for.** Task selection sorts by keyed
SHA-256 rather than a seeded RNG, so it survives interpreter changes. Scores
quantise at a fixed grid, so two validators summing in different float orders
emit identical vectors.

## Layout

```
microtensor/
├── core/       types, constants, canonical hashing, the gate, certificates
├── scoring/    metrics, geometric schedule, hysteresis, weight blending
├── chain/      the only place the Bittensor SDK is imported
├── harness/    engine contract, resource limits, the execution jail
├── envelope/   device profile, latency distributions, RSS sampler, profiler
├── registry/   signed manifests, digest-pinned fetch, LRU artifact cache
├── tasks/      corpus loading, deterministic per-round selection
├── store/      versioned sqlite state that survives a restart
├── update/     signed releases applied only between rounds
├── validator/  discover → evaluate → settle → the round loop
├── miner/      selfcheck, package, upload, publish
└── cli/        mt validator | miner | corpus | update | inspect

neurons/        validator.py, miner.py
deploy/         Dockerfiles and compose for both roles
```

## Development

```bash
pip install ".[validator,dev]"
PYTHONPATH=. pytest tests -q
ruff check . && mypy microtensor
```

The suite runs a **full round end to end** with no chain and no network:
commitment → manifest → fetch → jail → profile → gate → score → allocate →
blend → weight vector. It also covers a restart, a tampered artifact, an unsigned
manifest, and an unreachable source.

Code and tests carry no comments by design. The reasoning lives in
[docs/mechanism.md](docs/mechanism.md), where it can be argued with.
