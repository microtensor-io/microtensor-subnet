# Mechanism

Miners ship a specialised small model, a harness, a router and an escalation path as two Hugging Face repositories and commit the revision on chain with a fee. Each epoch, validators decrypt a hidden item bundle, run every submission inside a gVisor sandbox on their own GPUs, and score each part in its own pool against the arena's reference components. Scores are one sided lower bounds; a ladder keeps each miner's best per artifact; validators set weights themselves and the server publishes the consensus ledger.

The documentation lives on the site:

- https://www.microtensor.cloud/docs/certified-inference/getting-started/how-it-works
- https://www.microtensor.cloud/docs/certified-inference/getting-started/incentive-mechanism
- https://www.microtensor.cloud/docs/certified-inference/verification/sandbox-and-tests
- https://www.microtensor.cloud/docs/certified-inference/verification/scoring

Code map: `microtensor/chain` (commitments, epochs, weights), `microtensor/core/items.py` (bundles), `microtensor/artifact` (manifest and checks), `microtensor/validator` (the epoch pipeline), `microtensor/scoring` (bounds, pools, SQL grading), `microtensor/cli/submit.py` and `microtensor/cli/epoch.py`.
