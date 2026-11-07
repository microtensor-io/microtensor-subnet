# Mechanism

Miners ship a specialised small model, a harness, a router and an escalation path as two Hugging Face repositories and commit the revision on chain with a fee. Each epoch, validators decrypt a hidden item bundle, run every submission inside a locked down Docker container on their own GPUs (an internal network to the gateway only, CPU, memory and process limits, read only mounts, every capability dropped), and score each part in its own pool against the arena's reference components. Scores are one sided lower bounds; a ladder keeps each miner's best per artifact; validators set weights themselves and the server publishes the consensus ledger.

The documentation lives on the site:

- https://www.microtensor.cloud/docs/certified-inference/getting-started/how-it-works
- https://www.microtensor.cloud/docs/certified-inference/getting-started/incentive-mechanism
- https://www.microtensor.cloud/docs/certified-inference/verification/sandbox-and-tests
- https://www.microtensor.cloud/docs/certified-inference/verification/scoring

Serving (mechanism 0.8.0) is settled apart from the arena pools: once a UTC day the gateway's signed receipts are folded per operator card; a card scores its successful requests, 1.0 inside the arena's envelope and 0.5 outside, times its probe pass rate; team operators and every hotkey but the best under a coldkey score zero and burn; the first outside operator live on a newly certified image takes the arena's bounty share; the rebate caps the pool at 1/25 of the billing served. Validators check the receipts root and signatures before paying and burn the pool otherwise (`microtensor_subnet/serving/settle.py`, `pools.py`, `receipts.py`, `audit.py`).

Code map: `microtensor_subnet/chain` (commitments, epochs, weights), `microtensor_subnet/core/items.py` (bundles), `microtensor_subnet/artifact` (manifest and checks), `microtensor_subnet/validator` (the epoch pipeline), `microtensor_subnet/scoring` (bounds, pools, SQL grading), `microtensor_subnet/cli/submit.py` and `microtensor_subnet/cli/epoch.py`.
