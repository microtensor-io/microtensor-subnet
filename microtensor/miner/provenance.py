from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from microtensor.core.constants import (
    ALLOWED_BASE_MODELS,
    PROVENANCE_ENTITY,
    PROVENANCE_PROJECT,
    PROVENANCE_REQUIRED,
)
from microtensor.miner.config import MinerConfig
from microtensor.provenance.record import (
    CONFIG_BASE_MODEL,
    CONFIG_CLASS,
    CONFIG_CORPUS,
    CONFIG_TRACK,
    SUMMARY_DIGEST,
    SUMMARY_FINISHED,
    ProvenanceUnavailable,
    RunStore,
    Verdict,
    check,
)


class ProvenanceMissing(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Report:
    verdict: Verdict
    hotkey: str
    artifact_digest: str

    def render(self) -> str:
        run = self.verdict.run
        lines = [
            f"run          {run.path if run else '(none)'}",
            f"status       {'resolvable' if run else 'NOT FOUND'}",
        ]
        if run is not None:
            match = (
                "matches submitted artifact"
                if run.digest == self.artifact_digest
                else "MISMATCH"
            )
            base = str(run.config.get(CONFIG_BASE_MODEL, "(absent)"))
            listed = (
                "on allowlist"
                if not ALLOWED_BASE_MODELS or base in ALLOWED_BASE_MODELS
                else "NOT on allowlist"
            )
            finished = (
                datetime.fromtimestamp(run.finished_at, timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
                if run.finished_at
                else "(not recorded)"
            )
            lines += [
                f"digest       {match}",
                f"base model   {base}  ({listed})",
                f"finished     {finished}",
            ]
        lines.append(
            f"verdict      {'ADMISSIBLE' if self.verdict.admissible else 'REJECTED'}"
        )
        if not self.verdict.admissible:
            lines.append(f"reason       {self.verdict.reason}")
        return "\n".join(lines)


def init_snippet(config: MinerConfig, corpus_version: str = "<corpus digest>") -> str:
    return (
        "wandb.init(\n"
        f'    entity="{PROVENANCE_ENTITY}",\n'
        f'    project="{PROVENANCE_PROJECT}",\n'
        '    name="<your-hotkey-ss58>",\n'
        "    config={\n"
        f'        "{CONFIG_TRACK}": "{config.track}",\n'
        f'        "{CONFIG_CLASS}": "{config.hardware_class}",\n'
        f'        "{CONFIG_BASE_MODEL}": "<repo>@<sha>",\n'
        f'        "{CONFIG_CORPUS}": "{corpus_version}",\n'
        "    },\n"
        ")"
    )


def digest_snippet(artifact_digest: str) -> str:
    return (
        f'wandb.run.summary["{SUMMARY_DIGEST}"] = "{artifact_digest}"\n'
        f'wandb.run.summary["{SUMMARY_FINISHED}"] = int(time.time())\n'
        "wandb.finish()"
    )


def verify(
    store: RunStore,
    config: MinerConfig,
    hotkey: str,
    artifact_digest: str,
    *,
    commit_time: int = 0,
) -> Report:
    verdict = check(
        store.fetch(hotkey),
        hotkey=hotkey,
        artifact_digest=artifact_digest,
        track=config.track,
        hardware_class=config.hardware_class,
        commit_time=commit_time,
    )
    return Report(verdict=verdict, hotkey=hotkey, artifact_digest=artifact_digest)


def require(
    store: RunStore,
    config: MinerConfig,
    hotkey: str,
    artifact_digest: str,
    *,
    commit_time: int = 0,
) -> Report:
    if not PROVENANCE_REQUIRED:
        return Report(Verdict(True, "provenance is not required"), hotkey, artifact_digest)

    try:
        report = verify(store, config, hotkey, artifact_digest, commit_time=commit_time)
    except ProvenanceUnavailable as exc:
        raise ProvenanceMissing(
            f"the run store could not be reached, so your submission cannot be checked "
            f"before you commit it: {exc}"
        ) from exc

    if not report.verdict.admissible:
        raise ProvenanceMissing(
            f"{report.verdict.reason}; validators will reject this at discovery, so it "
            "is not being committed. Run `mt miner provenance` for the detail."
        )
    return report
