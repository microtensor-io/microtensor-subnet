# Compute mining on Microtensor

You put a GPU machine into the pool. It is verified, characterised and then
earns for being available and for the work placed on it. You run no models
yourself and you train nothing.

If you want to train models read [system_miner.md](system_miner.md). If you want
to serve them read [inference_miner.md](inference_miner.md). To choose between
the three, read [miner_setup.md](miner_setup.md).

Listing is free. There is no fee, no deposit and nothing to stake.

---

## 1 · Requirements

A machine failing any of these is not enrolled, and one that later stops meeting
a requirement drops out of rotation at the next deep pass.

| requirement | value | why |
|---|---|---|
| operating system | Ubuntu 22.04 or 24.04 | everything is tested against these |
| kernel | 5.19 minimum, 6.x preferred | container isolation over layered filesystems needs identity mapped mounts, added in 5.19 |
| architecture | x86_64 | required by the isolation runtime |
| GPU | on the accepted list, 8 GB or more, identified by UUID | the pool only characterises cards it can verify |
| driver | at or above the announced minimum | the driver runs privileged, so a hole there reaches every tenant |
| storage | 180 GB solid state minimum | below this the model cache, images and logs contend |
| storage quotas | enforceable by the docker storage driver | a tenant that can fill the host disk takes down every other tenant |
| runtime | docker-ce, containerd, the NVIDIA container toolkit, sysbox-ce 0.6.6 for rental | sysbox is what makes a stranger's code safe on your machine |
| network | public IPv4 with no carrier grade NAT, SSH reachable from validators, a reserved port range for tenants | the validator must reach the machine directly |
| exclusivity | nothing else running on the cards | anything the pool did not place is unauthorised |

Small models are memory bandwidth bound, so an older card with good bandwidth
serves them well and cheaply. You do not need a training card to qualify.

---

## 2 · Sign in and bind a hotkey

Open the Compute Pool page on the site. Connect a Bittensor wallet, pick the
hotkey and sign the nonce, then connect Discord once. The pool binds that
Discord account to the hotkey permanently and checks you are in the server.
Claims are refused until both are done.

The hotkey you claim with owns the rig and receives its earnings. Use one you
intend to keep, and not the one you mine or serve with.

---

## 3 · Install the agent

On the rig, as root:

```bash
curl -fsSL https://raw.githubusercontent.com/microtensor-io/microtensor-compute/main/agent/deploy/install.sh | sudo bash
```

The installer sets nothing up for you. It checks the machine against every
requirement above and refuses with the fix for each missing item. When the host
passes it pins the agent image to the digest the validators signed, writes the
compose file and environment under `/opt/rig-agent`, and starts three
containers.

| container | what it does |
|---|---|
| `agent` | the agent itself, with the agent API on port 8800 and SSH on port 2200 |
| `monitor` | the same image reading kernel messages for GPU faults, so a wedged agent cannot take monitoring down with it |
| `autoheal` | restarts a container marked unhealthy |

It enrols with the pool and prints a registration code. Read it again at any
time:

```bash
cd /opt/rig-agent && sudo docker compose exec agent rig-agent status
```

---

## 4 · Claim the rig

On the portal open Add rig, paste the code and press Add rig. The pool tells the
agent which hotkey is claiming, and the agent asks at the rig's own terminal.
Approve there:

```bash
cd /opt/rig-agent && sudo docker compose exec agent rig-agent approve <your hotkey>
```

Deny with `--deny`. The prompt expires after ten minutes. Only the person at the
keyboard can approve, which is what makes a leaked code useless.

The claim goes straight to validation. There is nothing to pay.

---

## 5 · The first deep pass

Within the hour a compute validator opens a session and runs its checks. The
result appears on the Validation page with each check and its value.

| check | what passes |
|---|---|
| host reality | the agent runs on a real host, not inside a container |
| GPU model and memory | every enrolled card is present, reported memory sits inside the model's window, no virtualised or partitioned slice |
| cryptographic challenge | the right answer, fast enough |
| throughput | the measured figure is consistent with the declared card |
| exclusivity | no GPU process the pool did not place |

A pass reads your rig's profile: its tier, and per kind of work whether it
qualifies and why not. A fail names the failure class and the seed, so you can
reproduce it yourself.

---

## 6 · Tiers and what they mean

Tiers are set by memory, because memory decides whether a workload fits at all.
Generation shows up separately in throughput, and therefore in what a machine
actually earns.

| tier | memory | weight |
|---|---|---|
| flagship | 80 GB and above | 4x |
| professional | 32 to 48 GB | 3x |
| standard | 20 to 24 GB | 2x |
| entry | 8 to 16 GB | 1x |

A rig's tier is the lowest tier among its accepted cards. A card under 8 GB is
not accepted for work. The current table with its memory windows is served at
`GET /v1/pool/hardware`.

---

## 7 · Work and pay

After the first pass the rig earns emission in full and takes low value work
while reliability builds. A day of clean checks opens the full priority ladder:
rental, inference, model shaping, then mining.

Earnings are ninety percent of what each job paid, settled every twenty four
hours. The compute pool takes ten percent of subnet emission, divided by
verified availability weighted by card class, with the larger share reserved for
machines actually running work rather than sitting idle. A machine that is
offline earns nothing for that period.

### Declining rental

Rental means a stranger's code runs on your hardware, inside a container with
its own user namespace. If you would rather not, turn rental off for the rig on
the portal. It costs you the highest value work in the ladder and a fraction of
the emission floor, and nothing else changes.

---

## 8 · Keeping it healthy

- **Keep it online.** A websocket ping every twenty seconds is the liveness
  signal. A drop longer than a minute marks the rig offline and its emission
  stops on that tick until a deep pass passes again.
- **Keep the driver current.** A cutoff is announced with a grace period, and a
  rig mid rental is exempt until the tenant leaves.
- **Let the agent update itself.** It applies only image digests signed by a
  compute validator, fetched every five minutes.
- **Read the logs** when something looks wrong: `sudo docker compose logs agent`
  and the append only event log at `/var/lib/rig-agent-logs/events.jsonl`.

### Removing a rig

A removed rig drains first, so a customer is never interrupted. Enrolling the
same card again later starts from enrolment.

---

## Where the code is

The agent, the compute validator and the challenge library live in
`microtensor-compute`. The pool itself is the compute module of the server. The
full compute pool documentation, including every accepted card and its memory
window, is at <https://microtensor.cloud/docs/compute>.
