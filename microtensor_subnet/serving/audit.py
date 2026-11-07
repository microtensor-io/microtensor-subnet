from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from microtensor_subnet.serving import receipts as receipt_rules
from microtensor_subnet.serving.client import ServerError

log = logging.getLogger("microtensor_subnet.serving.audit")

SETTLEMENT_PATH = "/v1/public/inference/settlement"
OPERATORS_PATH = "/v1/public/inference/operators"
RECEIPTS_PATH = "/v1/public/inference/receipts"
CATALOGUE_PATH = "/v1/public/inference/catalogue"
TIMEOUT_SECONDS = 60
COMPARED: Final[tuple[str, ...]] = (
    "score",
    "inside",
    "outside",
    "probes",
    "passes",
    "billing",
    "coldkey",
    "team",
)


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
    receipts: int = 0
    bad_receipts: int = 0
    root_ok: bool | None = None

    @property
    def holds(self) -> bool:
        return (
            not self.disagreements
            and not self.invented
            and not self.omitted
            and self.root_ok is not False
            and not self.bad_receipts
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "holds": self.holds,
            "checked": self.checked,
            "agreed": self.agreed,
            "disagreements": [str(d) for d in self.disagreements],
            "invented": list(self.invented),
            "omitted": list(self.omitted),
            "receipts": self.receipts,
            "bad_receipts": self.bad_receipts,
            "root_ok": self.root_ok,
        }


def _open(server: str, path: str, timeout: int, query: Mapping[str, Any] | None = None) -> str:
    url = urllib.parse.urljoin(server.rstrip("/") + "/", path.lstrip("/"))
    if query:
        url += "?" + urllib.parse.urlencode(dict(query))
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return str(answer.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ServerError(f"{exc.code}: could not read {path}") from exc
    except urllib.error.URLError as exc:
        raise ServerError(f"could not reach {url}: {exc.reason}") from exc
    except (OSError, ValueError) as exc:
        raise ServerError(f"could not read {url}: {exc}") from exc


def _get(server: str, path: str, timeout: int) -> dict[str, Any]:
    try:
        return dict(json.loads(_open(server, path, timeout)))
    except (TypeError, ValueError) as exc:
        raise ServerError(f"{path} did not answer with an object") from exc


def fetch(server: str, timeout: int = TIMEOUT_SECONDS) -> dict[str, Any]:
    return _get(server, SETTLEMENT_PATH, timeout)


def fetch_operators(server: str, timeout: int = TIMEOUT_SECONDS) -> list[dict[str, Any]]:
    found = _get(server, OPERATORS_PATH, timeout)
    return [dict(row) for row in found.get("operators") or [] if isinstance(row, Mapping)]


def parse_receipts(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            found = json.loads(line)
        except ValueError as exc:
            raise ServerError("the receipt export carries a line that is not json") from exc
        if isinstance(found, Mapping):
            rows.append(dict(found))
    return rows


def fetch_receipts(
    server: str, *, since: str, until: str, timeout: int = TIMEOUT_SECONDS
) -> list[dict[str, Any]]:
    return parse_receipts(_open(server, RECEIPTS_PATH, timeout, {"since": since, "until": until}))


def receipt_key(server: str, document: Mapping[str, Any], timeout: int = TIMEOUT_SECONDS) -> str:
    key = str(document.get("receipt_public_key") or "")
    if key:
        return key
    try:
        policy = _get(server, CATALOGUE_PATH, timeout).get("policy")
    except ServerError:
        return ""
    return str(policy.get("receipt_public_key") or "") if isinstance(policy, Mapping) else ""


def window_of(document: Mapping[str, Any]) -> tuple[str, str]:
    window = document.get("window")
    if isinstance(window, Mapping):
        return str(window.get("since") or ""), str(window.get("until") or "")
    return str(document.get("since") or ""), str(document.get("until") or "")


def receipts_of(server: str, document: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    if not document.get("receipts_root"):
        return None
    since, until = window_of(document)
    if not since or not until:
        raise ServerError("the settlement names a receipts root but no window")
    rows = fetch_receipts(server, since=since, until=until)
    key = receipt_key(server, document)
    for row in rows:
        row.setdefault("_key", key)
    return rows


def recompute(evidence: Mapping[str, Any]) -> list[Any]:
    from microtensor_subnet.serving.settle import settle_epoch

    rates = {str(k): int(v) for k, v in dict(evidence.get("rates") or {}).items()}
    rows = [dict(row) for row in evidence.get("rows") or []]
    found: list[Any] = settle_epoch(rows, rates=rates)
    return found


def _key(row: Mapping[str, Any]) -> str:
    head = f"{row.get('operator', '')}/{row.get('model', '')}"
    gpu = str(row.get("gpu", "") or "")
    return f"{head}/{gpu}" if gpu else head


def check(
    document: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]] | None = None
) -> Audit:
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
        for name in COMPARED:
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
    expected = str(document.get("receipts_root") or "")
    if expected and receipts is not None:
        key = next((str(r.get("_key") or "") for r in receipts if r.get("_key")), "")
        found.root_ok, found.bad_receipts, found.receipts = receipt_rules.check(
            receipts, public_key=key, expected_root=expected
        )
    return found


def audit(server: str) -> Audit:
    document = fetch(server)
    return check(document, receipts=receipts_of(server, document))
