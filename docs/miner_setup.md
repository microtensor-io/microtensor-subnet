# Mining on Microtensor

There are three ways to mine here, and they are separate. Different work,
different hardware, different hotkeys, different emission. Read this page to
find out which one you are, then follow that one's guide.

| | system miner | inference miner | compute miner |
|---|---|---|---|
| you provide | a certified inference system | serving capacity | a GPU machine |
| hardware | whatever you train on | a GPU, 8 GB or more | a card on the accepted list |
| the network measures | your artifact, on its own reference hardware | tokens you served, by verifying a sample | availability and isolation, by SSH and probes |
| you are paid for | where your system sits on the cost and quality frontier | billed tokens served | verified uptime and work done |
| settles every | round | serving epoch | compute epoch |
| capital at risk | registration only | posted collateral | registration only |
| your guide | [system_miner.md](system_miner.md) | [inference_miner.md](inference_miner.md) | [compute_miner.md](compute_miner.md) |

One machine can do all three under three hotkeys, and none of them knows about
the others. Most people do one.

---

## 1 · Which one are you

**You have a GPU and want to train.** You are a system miner. You compress a
frontier model into a specialist that fits a hardware class, publish the
artifact, and commit a pointer on chain once per round. Validators fetch it and
run it on their own certified hardware, so your machine is busy only while you
are training.

**You have a GPU and want it earning continuously without training.** You are
an inference miner. You post collateral, run a stock engine on certified
artifacts, and answer live traffic. You do not choose which models exist and you
do not pick which to hold; the network publishes what is certified and what is
wanted, and a planner on your machine decides. A card is required: systems are
ranked on a single CPU thread for determinism, but serving is judged on what a
client waits for, and your VRAM is the budget the planner spends.

**You have GPUs you want to rent out.** You are a compute miner. You put the
machine in the pool, it is verified and characterised, and it earns for being
available and for the work placed on it. You run no models yourself.

**You want to evaluate other people's work.** That is a validator, not a miner.
Read [validator_setup.md](validator_setup.md).

---

## 2 · How emission is divided

Emission splits before any miner is ranked. The three pools are sized
independently and a miner in one never competes with a miner in another.

```
compute   = compute_share * emission
remaining = emission - compute
available = (1 - reserve) * remaining
serving   = min(rebate * total_score, cap * available)
modelling = available - serving
```

| pool | current dial | who it pays |
|---|---|---|
| compute | 10 % of emission | compute miners |
| serving | up to 50 % of what remains, and only what billed traffic earned | inference miners |
| modelling | everything left | system miners |

Two properties hold by construction rather than by policy.

**With no billed traffic the serving pool is empty.** It is sized by what
serving actually earned, so nothing is set aside waiting for a serving layer
that has not arrived. That is the state today: the launch is meter only on free
allowance, so modelling and compute take everything.

**Serving can never starve modelling.** It is capped as a fraction of what is
available. A network that stopped paying for new models would stop having
anything worth serving.

The reserve is an operator lever that holds emission back entirely, currently
set to zero.

### Inside the modelling pool

Split across competitions by their published share, then paid by rank within
each competition. The top eight places earn on a geometric curve that decays by
15 % a rank, so eighth place still earns about a third of first. Ranks below
eight earn nothing that round.

```bash
mt inspect tracks
```

### Inside the serving pool

Per inference miner and model, over one serving epoch. Your score is billed
tokens valued at a reference rate the network sets per model, not at what the
client paid, so two miners doing identical work score identically. Four gates
apply, and each fails only when the evidence is statistically clear. A failed
gate forfeits that epoch and excludes nobody.

### Inside the compute pool

By verified availability weighted by the class of card, with a large share
reserved for machines actually running work rather than sitting idle. A machine
that is offline earns nothing for that period.

---

## 3 · Registration

Every role registers a hotkey on the subnet in the same way.

```bash
btcli subnet register --netuid 576 --subtensor.network test
```

| you also need | system miner | inference miner | compute miner |
|---|---|---|---|
| registration burn | yes | yes | yes |
| a submission fee per round | yes | no | no |
| posted collateral | no | yes | no |
| a listing fee | no | no | no, free since September 2026 |

Use a different hotkey per role. They are measured separately, scored
separately and paid separately, and mixing them makes none of that work better.

---

## 4 · What the network guarantees you

The same three things, whichever role you take.

**Measurement is not self reported.** A system miner's envelope is measured by
validators on certified hardware. An inference miner's tokens are verified
against the certified artifact by validators holding their own copy. A compute
miner's availability is probed, not claimed.

**A fault of the network never scores against you.** A validator that cannot
reach your artifact abstains rather than scoring you zero. A serving sample that
cannot be evaluated is unproven and counts against nobody. A compute probe that
fails on the pool's side is not a strike.

**Scoring never bans.** A failed epoch or a missed round costs you that period
and nothing more. You start the next one clean.

---

## Where to go next

| | |
|---|---|
| [system_miner.md](system_miner.md) | Train, package, publish and commit, round by round |
| [inference_miner.md](inference_miner.md) | Register, post collateral, serve and get verified |
| [compute_miner.md](compute_miner.md) | Enrol a GPU machine in the pool |
| [validator_setup.md](validator_setup.md) | Evaluate submissions and verify serving |
| [mechanism.md](mechanism.md) | The scoring rules in full |
