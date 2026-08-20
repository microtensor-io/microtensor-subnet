from __future__ import annotations

import html
import logging
from collections.abc import Mapping, Sequence
from typing import Any

log = logging.getLogger("microtensor.coordinator.dashboard")

GATES = ("size", "memory", "latency", "declaration", "provenance", "base model")


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _num(value: Any, digits: int = 3, dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return dash


def _int(value: Any, dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return dash


def headline(
    round_index: int,
    reported: Sequence[Mapping[str, Any]],
    settlement: Mapping[str, Any] | None,
    summary: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """The row a reader checks first.

    The counterpart of a training dashboard's loss and throughput line, in the
    terms this network actually settles on.
    """
    frontier = list((settlement or {}).get("frontier", ()))
    admitted = [row for row in frontier if float(row.get("share", 0.0)) > 0.0]
    training = [r for r in reported if str(r.get("phase")) == "training"]

    best = max((float(r.get("quality", 0.0)) for r in admitted), default=None)
    cheapest = min((float(r.get("expected_ms", 0.0)) for r in admitted), default=None)
    resolve = [float(s.get("median_resolve_rate", 0.0)) for s in summary if s.get("member_count")]

    return [
        ("Round", _int(round_index)),
        ("Miners training", _int(len(training))),
        ("Systems submitted", _int(len(frontier))),
        ("Systems admitted", _int(len(admitted))),
        ("Frontier size", _int(sum(int(s.get("member_count", 0)) for s in summary))),
        ("Best quality", _num(best, 4)),
        ("Lowest cost", f"{_num(cheapest, 1)} ms" if cheapest is not None else "—"),
        ("Median resolve", _num(sum(resolve) / len(resolve), 3) if resolve else "—"),
    ]


def composition(hardware: Sequence[Mapping[str, Any]]) -> list[tuple[str, int, int, float | None]]:
    """Active miners by accelerator, with derived throughput.

    The most informative single chart a training dashboard carries, and the
    reason hardware is collected at all.
    """
    from microtensor.miner.telemetry import tflops_for

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in hardware:
        groups.setdefault(str(row.get("gpu_name") or "unknown"), []).append(row)

    out = []
    for name, rows in groups.items():
        cards = sum(int(r.get("gpu_count") or 0) for r in rows)
        out.append((name, len(rows), cards, tflops_for(name, cards)))
    return sorted(out, key=lambda entry: (-entry[1], entry[0]))


def _table(columns: Sequence[str], rows: Sequence[Sequence[str]], empty: str) -> str:
    if not rows:
        return f'<p class="empty">{_esc(empty)}</p>'

    head = "".join(f"<th>{_esc(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def reported_table(rows: Sequence[Mapping[str, Any]], hardware: Sequence[Mapping[str, Any]]) -> str:
    from microtensor.miner.telemetry import tflops_for

    by_hotkey = {str(h.get("hotkey")): h for h in hardware}

    built = []
    for row in rows:
        hotkey = str(row.get("hotkey", ""))
        gear = by_hotkey.get(hotkey, {})
        cards = int(gear.get("gpu_count") or 0)
        built.append(
            [
                f'<code>{_esc(hotkey[:12])}</code>',
                _esc(gear.get("gpu_name") or "—"),
                _int(cards) if cards else "—",
                _num(tflops_for(str(gear.get("gpu_name") or ""), cards), 0),
                f'<span class="phase">{_esc(row.get("phase"))}</span>',
                _int(row.get("last_epoch")),
                _num(row.get("loss"), 4),
                _num(row.get("throughput"), 1),
                _num(row.get("mfu"), 3),
                _int(row.get("elapsed_s")),
                _int(row.get("last_block")),
            ]
        )

    return _table(
        (
            "Hotkey", "GPU", "Count", "TFLOPS", "Phase", "Epoch", "Loss",
            "Throughput", "MFU", "Elapsed", "Last block",
        ),
        built,
        "No miner has reported yet.",
    )


def measured_table(settlement: Mapping[str, Any] | None) -> str:
    rows = list((settlement or {}).get("frontier", ()))
    ordered = sorted(rows, key=lambda r: -float(r.get("share", 0.0)))

    built = []
    for row in ordered:
        contribution = dict(row.get("contribution") or {})
        split = " · ".join(
            f"{role} {_num(value, 2)}" for role, value in sorted(contribution.items())
        )
        built.append(
            [
                f'<code>{_esc(str(row.get("miner", ""))[:12])}</code>',
                f'<code>{_esc(str(row.get("system", ""))[:16])}</code>',
                _num(row.get("quality"), 4),
                f'{_num(row.get("expected_ms"), 1)} ms',
                _num(row.get("share"), 4),
                _esc(split) or "—",
            ]
        )

    return _table(
        ("Miner", "System", "Quality", "Cost", "Exclusive HV", "Contribution"),
        built,
        "Nothing has settled yet.",
    )


def gate_matrix(settlement: Mapping[str, Any] | None) -> str:
    """One row per system, a pass or a fail per check.

    The single most useful view for someone who has just been rejected, and it
    needs no data the settlement does not already publish.
    """
    published = settlement or {}
    unscored = set(published.get("unscored", ()))

    built = []
    for row in published.get("frontier", ()):
        digest = str(row.get("system", ""))
        admitted = float(row.get("share", 0.0)) > 0.0 and digest not in unscored
        mark = '<span class="pass">pass</span>' if admitted else '<span class="fail">fail</span>'
        built.append(
            [f'<code>{_esc(digest[:16])}</code>', *([mark] * len(GATES))]
        )

    return _table(("System", *GATES), built, "No system has been gated yet.")


def movement(summary: Sequence[Mapping[str, Any]]) -> str:
    """Frontier advance, one point per round.

    A training network draws a loss curve descending. A frontier is a surface,
    and its total covered ground rising is the same claim about progress made
    on both axes at once.
    """
    points = [(int(s["round_index"]), float(s.get("hv_total", 0.0))) for s in summary]
    if len(points) < 2:
        return '<p class="empty">Two rounds are needed before movement can be drawn.</p>'

    top = max(value for _, value in points) or 1.0
    width, height, pad = 640, 160, 18
    span = max(1, points[-1][0] - points[0][0])

    def x(index: int) -> float:
        return pad + (index - points[0][0]) / span * (width - pad * 2)

    def y(value: float) -> float:
        return height - pad - (value / top) * (height - pad * 2)

    line = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in points)
    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
        f'aria-label="Frontier advance across rounds">'
        f'<polyline points="{line}" />'
        f"</svg>"
    )


STYLE = """
:root { --ink:#101018; --soft:#6b6b78; --line:#e4e4ea; --pass:#127a4a; --fail:#b0304a; }
* { box-sizing:border-box; }
body { margin:0; padding:32px; font:14px/1.5 ui-sans-serif,system-ui,sans-serif;
       color:var(--ink); background:#fff; }
h1 { margin:0 0 4px; font-size:20px; }
h2 { margin:32px 0 4px; font-size:15px; }
p.note { margin:0 0 16px; color:var(--soft); font-size:12px; }
p.empty { margin:8px 0; color:var(--soft); font-size:12px; }
.headline { display:flex; flex-wrap:wrap; gap:24px; padding:16px 0;
            border-block:1px solid var(--line); }
.headline div { min-width:96px; }
.headline dt { color:var(--soft); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
.headline dd { margin:2px 0 0; font-size:20px; font-weight:600; font-variant-numeric:tabular-nums; }
table { width:100%; border-collapse:collapse; margin-top:8px; font-size:12px; }
th { text-align:left; padding:6px 10px; color:var(--soft); font-weight:600;
     text-transform:uppercase; font-size:10px; letter-spacing:.06em;
     white-space:nowrap; }
td { padding:6px 10px; border-top:1px solid var(--line); font-variant-numeric:tabular-nums;
     white-space:nowrap; }
code { font:11px ui-monospace,monospace; }
.phase { font:11px ui-monospace,monospace; }
.pass { color:var(--pass); } .fail { color:var(--fail); }
.chart { width:100%; height:auto; margin-top:8px; }
.chart polyline { fill:none; stroke:var(--ink); stroke-width:1.5; }
.wrap { overflow-x:auto; }
"""


def render(
    round_index: int,
    reported: Sequence[Mapping[str, Any]],
    hardware: Sequence[Mapping[str, Any]],
    settlement: Mapping[str, Any] | None,
    summary: Sequence[Mapping[str, Any]],
) -> str:
    """One page, from what the coordinator already stores.

    Two tables kept visually distinct, because they answer different questions
    and carry different weight: what miners say they are doing, and what
    validators measured.
    """
    cells = "".join(
        f"<div><dt>{_esc(label)}</dt><dd>{value}</dd></div>"
        for label, value in headline(round_index, reported, settlement, summary)
    )

    gear = "".join(
        f"<tr><td>{_esc(name)}</td><td>{_int(miners)}</td>"
        f"<td>{_int(cards)}</td><td>{_num(tflops, 0)}</td></tr>"
        for name, miners, cards, tflops in composition(hardware)
    )
    gear_table = (
        f"<table><thead><tr><th>Accelerator</th><th>Miners</th><th>Cards</th>"
        f"<th>TFLOPS</th></tr></thead><tbody>{gear}</tbody></table>"
        if gear
        else '<p class="empty">No hardware reported.</p>'
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Microtensor round {round_index}</title>
<style>{STYLE}</style></head><body>
<h1>Round {round_index}</h1>
<p class="note">Reported figures come from miners and are observational. Measured
figures come from validators on certified hardware.</p>

<dl class="headline">{cells}</dl>

<h2>Reported: who is working</h2>
<p class="note">Self reported by miners. Never enters a certificate or a score.</p>
<div class="wrap">{reported_table(reported, hardware)}</div>

<h2>Network composition</h2>
<p class="note">Self reported. TFLOPS is derived from the accelerator name, not
accepted from the miner.</p>
<div class="wrap">{gear_table}</div>

<h2>Measured: who is winning</h2>
<p class="note">Computed by validators on certified reference hardware.</p>
<div class="wrap">{measured_table(settlement)}</div>

<h2>Frontier advance</h2>
{movement(summary)}

<h2>Gate matrix</h2>
<div class="wrap">{gate_matrix(settlement)}</div>
</body></html>
"""
