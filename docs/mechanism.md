# Mechanism

Miners ship a specialised small model, a harness, a router and an escalation path as two Hugging Face repositories and commit the revision on chain with a fee. Each epoch, validators decrypt a hidden item bundle, run every submission inside a locked down Docker container on their own GPUs (an internal network to the gateway only, CPU, memory and process limits, read only mounts, every capability dropped), and score each part in its own pool against the arena's reference components. Scores are one sided lower bounds; a ladder keeps each miner's best per artifact; validators set weights themselves and the server publishes the consensus ledger.

The documentation lives on the site:

- https://www.microtensor.cloud/docs/certified-inference/getting-started/how-it-works
- https://www.microtensor.cloud/docs/certified-inference/getting-started/incentive-mechanism
- https://www.microtensor.cloud/docs/certified-inference/verification/sandbox-and-tests
- https://www.microtensor.cloud/docs/certified-inference/verification/scoring

Code map: `microtensor_subnet/chain` (commitments, epochs, weights), `microtensor_subnet/core/items.py` (bundles), `microtensor_subnet/artifact` (manifest and checks), `microtensor_subnet/validator` (the epoch pipeline), `microtensor_subnet/scoring` (bounds, pools, SQL grading), `microtensor_subnet/cli/submit.py` and `microtensor_subnet/cli/epoch.py`.
