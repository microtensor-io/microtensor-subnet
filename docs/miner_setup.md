# Mining on Microtensor

A submission is two Hugging Face repositories: the artifact (`<user>/microtensor_<arena>`: harness, router, escalation, manifest) and the model (`<user>/mt_slm_<arena>`: weights and the training record). Validators fetch both at the revision you commit on chain and run every part on their own GPUs.

Start from the template: https://github.com/microtensor-io/microtensor-miner-template. Its README walks through the adapter for your arena's task, the two layouts, `mt submit manifest`, `mt submit validate`, `mt submit pay` and `mt submit commit`.

The full guide, the sandbox contract and the scoring rules are on the site: https://www.microtensor.cloud/docs/certified-inference/guides/mining

## Training playbook

Validators score your weights under the arena's own prompt, and provenance fails a model that is still the base or matches a public checkpoint, so prompting alone never certifies. The order that works:

1. Baseline: grade each allowed base on the public sample items with the arena's grader.
2. Harness: set the adapter prompt, the verifier and retries, measure again.
3. Distil: frontier answers with reasoning on schemas and question shapes like the arena's, filtered by the grader, fine tuned from the base (LoRA or full) and recorded under `training/`.
4. Reinforce: reward a query that returns the reference result on every database state; or preference pairs of a right teacher answer against a wrong draft.
5. Check: hold out whole domains, rename and revalue as the twin rule does, measure latency on a 24 GB card, train the router on the result.

Items are fresh every epoch and twins deduct memorisation, so train the skill, not the sample databases. Full page: https://www.microtensor.cloud/docs/certified-inference/guides/training
