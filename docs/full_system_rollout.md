# Full system rollout

Current arenas keep running unchanged. A system arena opens only at a round boundary, announced one round ahead, and nothing below changes a live arena's scoring until then.

## Already on main

| Piece | PRs |
|---|---|
| Manifest v2, escalation allowlist, harness package scan | #122 |
| Signed traces | #123 |
| Archive at submission, escalation mirrors | #124 |
| Harness runtime, hosted systems, live runner | #125 |
| Verification: replay, router, escalation, archive rerun | #126 |
| System scoring: quality, small model alone, escalation, waste, misses, calibration, cost | #127 |
| Router training, harness search, full system simulation, playbook | #128 |
| Routine and unusual task profiles | #130 |
| Validator evaluation of full systems | #131 |
| Live escalation rows | #132 |
| Serving certified systems from the archive | #133 |
| Output contract, input formats, two stage choices | #113, #119, #118 |

## Before the first system arena opens

1. **Boundary merges and fleet update.** Merge #107 with its stack (#108, #111, #112, #120, #129), #109, #114 and #115. Update the coordinator and every validator together, and bump `MECHANISM_REF` on the server.
2. **Server and gateway.** Merge and deploy server #90 (escalation allowlist) and #91 (escalations view), and gateway #9 (task route). All three wait on GitHub Actions billing.
3. **Choose the arena.** Add a track with `full_system=True`, enabled with its emission share, in the boundary release.
4. **Arena dials on the server.**
   - `escalation_models`: each allowed model at a pinned revision with its price per million tokens.
   - `reference_cost_ms` in micro USD per task (a system arena's cost unit).
   - `quality_floor` from `mt coordinator floor` on the base model.
   - `target_quality` and `target_model` for the outside model line.
5. **Mirror every escalation model once:** `mt archive mirror --model org/name --revision <sha>`.
6. **Corpus.** Withheld routine and unusual tasks, with at least 20% unusual, passing `mt corpus check`, and teacher data published for the train split.
7. **Validators.** Set `MT_GATEWAY_URL` and `MT_GATEWAY_SECRET`, and confirm the jail runs a GGUF replay on the Linux host.
8. **Coordinator box.** Run `mt archive intake --db <coordinator.sqlite> --watch 300 --reveals --org <private org>` so every submission is copied as it is committed.
9. **Announce one round ahead** (draft below), then open the round the usual way: server open with the config, coordinator open, anchor.

## Dials to confirm

| Dial | Value | Where |
|---|---|---|
| Reference CPU price for the small model | $0.04 per CPU hour | `scoring/system.py` |
| Replay margin tolerance | 0.5 logits | `validator/verify_system.py` |
| Confidence tolerance | 0.02 | `validator/verify_system.py` |
| Traces verified per system | 8, keyed on the round seed | `validator/verify_system.py` |
| Unusual share of scored tasks | at least 20% | `tasks/corpus.py` |
| Live rows per system | 500 | `validator/evaluate.py` |

## Announcement draft

> **Full system arenas open at round N.**
>
> From round N, the new system arenas take a full system, not a single model: your small specialist model, the harness around it, a router, and an escalation model chosen from the arena's allowlist. You host it through the dial out agent; no public IP is needed.
>
> Validators send withheld tasks to your system during the round and check every answer: your small model is replayed from the archive, your router's decisions are recomputed from its declared rule, and escalations are charged at the published price. You are ranked on end to end quality against total cost, and a system that fails a check earns nothing that round.
>
> Your submission is archived when you commit it, and by submitting you license us to host and serve it. Existing arenas run unchanged.
>
> Start with `docs/full_system_playbook.md`: teacher data, `train_specialist.py`, `optimise_harness.py`, `train_router.py` and `mt miner simulate` are ready now.
