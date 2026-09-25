from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from microtensor.serving.client import ServerError

log = logging.getLogger("microtensor.serving.audit")

SETTLEMENT_PATH = "/v1/public/inference/settlement"
TIMEOUT_SECONDS = 60


@dataclass(frozen=True, slots=True)
class Disagreement:
    operator: str
    model: str
    field: str
    published: Any
    recomputed: Any

    def __str__(self) -> str:
        return (
            f"{self.operator[:12]}/{self.model} {self.field}: "
            f"published {self.published}, recomputed {self.recomputed}"
        )


@dataclass(slots=True)
class Audit:
    checked: int = 0
    agreed: int = 0
    disagreements: list[Disagreement] = field(default_factory=list)
    invented: list[str] = field(default_factory=list)
    omitted: list[str] = field(default_factory=list)

    @property
    def holds(self) -> bool:
        return not self.disagreements and not self.invented and not self.omitted

    def to_dict(self) -> dict[str, Any]:
        return {
            "holds": self.holds,
            "checked": self.checked,
            "agreed": self.agreed,
            "disagreements": [str(d) for d in self.disagreements],
            "invented": list(self.invented),
            "omitted": list(self.omitted),
        }


def fetch(server: str, timeout: int = TIMEOUT_SECONDS) -> dict[str, Any]:
    url = urllib.parse.urljoin(server.rstrip("/") + "/", SETTLEMENT_PATH.lstrip("/"))
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return dict(json.loads(answer.read().decode("utf-8")))
    except urllib.error.HTTPError as exc:
        raise ServerError(f"{exc.code}: could not read the settlement") from exc
    except urllib.error.URLError as exc:
        raise ServerError(f"could not reach {url}: {exc.reason}") from exc
    except (OSError, ValueError) as exc:
        raise ServerError(f"could not read {url}: {exc}") from exc


def recompute(evidence: Mapping[str, Any]) -> list[Any]:
    from microtensor.serving.settle import settle_epoch

    rates = {str(k): int(v) for k, v in dict(evidence.get("rates") or {}).items()}
    thresholds = {
        str(name): Fraction(int(pair[0]), int(pair[1]))
        for name, pair in dict(evidence.get("thresholds") or {}).items()
        if isinstance(pair, Sequence) and len(pair) == 2
    }
    rows = [dict(row) for row in evidence.get("rows") or []]
    found: list[Any] = settle_epoch(rows, rates=rates, thresholds=thresholds)
    return found


def _key(row: Mapping[str, Any]) -> str:
    return f"{row.get('operator', '')}/{row.get('model', '')}"


def check(document: Mapping[str, Any]) -> Audit:
    evidence = dict(document.get("evidence") or {})
    published = {_key(row): dict(row) for row in document.get("settled") or []}

    try:
        ours = {_key(row.to_dict()): row.to_dict() for row in recompute(evidence)}
    except Exception as exc:
        raise ServerError(f"the published evidence does not settle: {exc}") from exc

    found = Audit()
    for key in sorted(set(published) | set(ours)):
        if key not in ours:
            found.invented.append(key)
            continue
        if key not in published:
            found.omitted.append(key)
            continue
        found.checked += 1
        theirs, mine = published[key], ours[key]
        operator = str(theirs.get("operator", ""))
        model = str(theirs.get("model", ""))
        clean = True
        for name in ("score", "tokens", "requests", "gate_reason"):
            if theirs.get(name) != mine.get(name):
                clean = False
                found.disagreements.append(
                    Disagreement(
                        operator=operator,
                        model=model,
                        field=name,
                        published=theirs.get(name),
                        recomputed=mine.get(name),
                    )
                )
        if clean:
            found.agreed += 1
    return found


def audit(server: str) -> Audit:
    return check(fetch(server))
