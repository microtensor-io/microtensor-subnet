# Mining on Microtensor

A submission is two Hugging Face repositories: the artifact (`<user>/microtensor_<arena>`: harness, router, escalation, manifest) and the model (`<user>/mt_slm_<arena>`: weights and the training record). Validators fetch both at the revision you commit on chain and run every part on their own GPUs.

Start from the template: https://github.com/microtensor-io/microtensor-miner-template. Its README walks through the adapter for your arena's task, the two layouts, `mt submit manifest`, `mt submit validate`, `mt submit pay` and `mt submit commit`.

The full guide, the sandbox contract and the scoring rules are on the site: https://www.microtensor.cloud/docs/certified-inference/guides/mining
