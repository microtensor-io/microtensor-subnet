# Full system playbook

A full system is a small model, the harness around it, a router and an escalation model. You host it, validators test it live on withheld tasks, and it earns on end to end quality against total cost. This playbook is the order that works and the numbers behind it.

## Why a system beats a single model

| Finding | Number | Source |
|---|---|---|
| An adapted harness lifts a small model on business tasks | about 30 to about 80 | CMU harness paper, arXiv 2607.08938 |
| The same system against a frontier model | 90% of its quality at 4% of its cost | CMU harness paper |
| Scoring only the allowed answers and training honest confidence | small models become fast, always valid and trustworthy | Subset, Jev |
| Reading a shared document once for every question | 3.4 to 3.9 times faster | our decide mode, R1 |
| A trained router against always escalating (simulation, step 6) | 0.978 against 0.955 quality at 37% of the cost | `scoring/system.py` |

The router is what turns confidence into money. It sends only the requests the small model will get wrong to the escalation model, so you pay for the large model only when it changes the answer.

## The order

1. **Pick the base.** Choose an allowlisted base that fits the arena's hardware class. Customers run your small model locally, so it has to fit.
2. **Distil from the teacher.** Every corpus publishes frontier teacher data for its train split (`scripts/generate_reference_completions.py`). Fine tune on it with `scripts/train_specialist.py`. Choice arenas train on answer probabilities; everything else trains on teacher completions.
3. **Reward being right and knowing it.** Run the RLCR stage of `train_specialist.py`: it rewards correct answers and penalises confidence that does not match correctness. A model that is right but unsure, or wrong but sure, loses reward.
4. **Search the harness.** `scripts/optimise_harness.py` runs your small model over the train split, shows its failures to a frontier model, and keeps the prompts that beat their parents, on a Pareto front across tasks. Tools and hooks stay yours; the search rewrites instructions.
5. **Fit the router.** `scripts/train_router.py` records your small model's own results on the train split with every permitted feature (confidence, entropy, input length, how typical the input is, harness errors) and fits a short rule that trades quality against escalations. Typicality is computed out of fold, so the router learns what an unusual input looks like.
6. **Calibrate and quantise.** Fold temperature into the file (`scripts/fold_temperature.py`) so the confidence validators recompute is already calibrated. Quantise with the output layer kept at Q8 (`--output-tensor-type q8_0`, or `--token-embedding-type q8_0` for tied embeddings).
7. **Simulate the whole system.** `mt miner simulate --escalation-url URL` runs the small model, harness, router and escalation locally over the train split and prints every trace with the scores validators will compute: end to end quality, the small model alone, escalation rate, waste, misses, calibration and cost.
8. **Package and host.** Write `system.json` at schema version 2 (model, harness with its runtime, router with its features, escalation model pinned to a revision, endpoint), then serve it through the dial out agent. No public IP is needed.
9. **Serve it to customers.** Once certified, inference miners serve your system under the arena's catalogue name on GPUs (`inference_miner.md`, section 7a), and we can serve it from the archive after you stop.

## What validators check

- **Your small model is the archived one.** They replay sampled steps on the archived GGUF on CPU; a token more than 0.5 logits below the model's own choice, or a confidence off by more than 0.02, fails the round.
- **Your router follows its rule.** They recompute the features from the verified small model output and the decision from your declared rule.
- **Your escalation is honest.** Only your declared, allowlisted model, charged at its published price.
- **The archive reproduces the live run.** Your archived harness, router and small model are rerun on sampled tasks; prompts, tokens, decisions and final answers must match.
- **Your harness stays inside its package.** No URLs, no network or process modules, no `eval`.
- **Your system is fast enough.** Validators time every request themselves; a system whose end to end p95 is over the arena's latency ceiling earns nothing that round.

A system that fails any check is not certified and earns nothing that round.

## Mistakes that cost emission

- **Escalating everything.** Quality rises, cost rises faster, and a trained router beats you on both.
- **Never escalating.** You keep every error the small model makes; misses are reported on your card.
- **Confidence that lies.** It fails replay, and a router fed on it escalates the wrong requests.
- **A harness that calls out.** It is refused at admission.
