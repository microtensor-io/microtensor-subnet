from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Final

from microtensor.core.constants import ALLOWED_ROUTER_FEATURES, HOST_PROFILE
from microtensor.core.escalation import REVISION
from microtensor.core.hashing import canonical_hash
from microtensor.core.protocol import Role


class SystemManifestError(ValueError):
    pass


FULL_SYSTEM: Final[int] = 2
HARNESS_SDK_VERSIONS: Final[frozenset[str]] = frozenset({"1.0.0"})
_IMAGE: Final[re.Pattern[str]] = re.compile(r"^image:[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}$")
_SDK: Final[re.Pattern[str]] = re.compile(r"^sdk:(\d+\.\d+\.\d+)$")


def _contained(path: str, what: str) -> None:
    parts = PurePosixPath(path).parts
    if not path or path.startswith("/") or ".." in parts:
        raise SystemManifestError(f"the {what} path {path!r} must be relative and contained")


@dataclass(frozen=True, slots=True)
class HarnessRef:
    package_digest: str
    runtime: str
    path: str = "harness"

    def __post_init__(self) -> None:
        if not self.package_digest.startswith("sha256:"):
            raise SystemManifestError("the harness carries no package digest")
        sdk = _SDK.match(self.runtime)
        if sdk is None and not _IMAGE.match(self.runtime):
            raise SystemManifestError(
                "the harness runtime must be a pinned container image "
                "(image:<name>@sha256:<digest>) or a harness SDK version (sdk:X.Y.Z)"
            )
        if sdk is not None and sdk.group(1) not in HARNESS_SDK_VERSIONS:
            raise SystemManifestError(
                f"harness SDK {sdk.group(1)} is not supported; use one of "
                f"{sorted(HARNESS_SDK_VERSIONS)}"
            )
        _contained(self.path, "harness")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EscalationRef:
    model: str
    revision: str

    def __post_init__(self) -> None:
        if "/" not in self.model:
            raise SystemManifestError(
                f"the escalation model {self.model!r} is not an org/name repository"
            )
        if not REVISION.match(self.revision):
            raise SystemManifestError("pin the escalation model to a full commit revision")

    @property
    def key(self) -> str:
        return f"{self.model}@{self.revision}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EndpointRef:
    worker: str
    name: str

    def __post_init__(self) -> None:
        if not self.worker or not self.name:
            raise SystemManifestError(
                "the endpoint names the dial out worker and the name the system is served under"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_PATHS: dict[Role, str] = {
    Role.FRONT: "front",
    Role.ROUTER: "router.json",
    Role.SPECIALIST: "specialist",
}


@dataclass(frozen=True, slots=True)
class ComponentRef:
    role: Role
    artifact_digest: str
    placement: str
    path: str = ""
    base_model: str = ""

    def __post_init__(self) -> None:
        if not self.artifact_digest:
            raise SystemManifestError(
                f"the {self.role.value} component carries no artifact digest"
            )
        if not self.placement:
            raise SystemManifestError(f"the {self.role.value} component declares no placement")
        if self.path:
            parts = PurePosixPath(self.path).parts
            if self.path.startswith("/") or ".." in parts:
                raise SystemManifestError(
                    f"the {self.role.value} path {self.path!r} must be relative and contained"
                )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        return d


@dataclass(frozen=True, slots=True)
class SystemManifest:
    front: ComponentRef
    router: ComponentRef | None = None
    specialist: ComponentRef | None = None
    router_features: tuple[str, ...] = ()
    schema_version: int = 1
    harness: HarnessRef | None = None
    escalation: EscalationRef | None = None
    endpoint: EndpointRef | None = None

    def __post_init__(self) -> None:
        if self.schema_version >= FULL_SYSTEM:
            missing = [
                part
                for part, value in (
                    ("harness", self.harness),
                    ("router", self.router),
                    ("escalation model", self.escalation),
                    ("endpoint", self.endpoint),
                )
                if value is None
            ]
            if missing:
                raise SystemManifestError(
                    "a full system declares a model, harness, router and escalation model, "
                    f"served from an endpoint; missing: {', '.join(missing)}"
                )
            if self.specialist is not None:
                raise SystemManifestError(
                    "a full system escalates to an allowlisted open model, "
                    "not a specialist artifact"
                )
            if not self.front.base_model:
                raise SystemManifestError("the small model must pin its base model")
        if self.front.role is not Role.FRONT:
            raise SystemManifestError("the front slot must carry a front component")
        if self.router is not None and self.router.role is not Role.ROUTER:
            raise SystemManifestError("the router slot must carry a router component")
        if self.specialist is not None and self.specialist.role is not Role.SPECIALIST:
            raise SystemManifestError("the specialist slot must carry a specialist component")

        if self.schema_version < FULL_SYSTEM and (self.router is None) != (
            self.specialist is None
        ):
            raise SystemManifestError(
                "a router and a specialist are jointly optional; declare both or neither"
            )

        if self.specialist is not None and self.specialist.placement != HOST_PROFILE:
            raise SystemManifestError(
                f"the specialist is placed on {self.specialist.placement!r}, "
                f"not the host profile {HOST_PROFILE!r}"
            )

        unpermitted = [f for f in self.router_features if f not in ALLOWED_ROUTER_FEATURES]
        if unpermitted:
            raise SystemManifestError(
                f"router declares features that are not permitted: {sorted(unpermitted)}"
            )
        if self.router is not None and not self.router_features:
            raise SystemManifestError("a router must declare the features it reads")
        if self.router is None and self.router_features:
            raise SystemManifestError("router features are declared without a router")
        if len(set(self.router_features)) != len(self.router_features):
            raise SystemManifestError("a router must not declare the same feature twice")

        digests = [c.artifact_digest for c in self.components]
        if len(set(digests)) != len(digests):
            raise SystemManifestError("a component digest appears more than once in one manifest")

    @property
    def components(self) -> tuple[ComponentRef, ...]:
        return tuple(c for c in (self.front, self.router, self.specialist) if c is not None)

    @property
    def degenerate(self) -> bool:
        return self.router is None and self.specialist is None and self.escalation is None

    @property
    def full(self) -> bool:
        return self.schema_version >= FULL_SYSTEM

    def component(self, role: Role) -> ComponentRef | None:
        return {
            Role.FRONT: self.front,
            Role.ROUTER: self.router,
            Role.SPECIALIST: self.specialist,
        }[role]

    def __iter__(self) -> Iterator[ComponentRef]:
        return iter(self.components)

    def fits_class(self, hardware_class: str) -> tuple[bool, str]:
        if self.front.placement != hardware_class:
            return False, (
                f"the front is placed on {self.front.placement!r}, "
                f"not the competition class {hardware_class!r}"
            )
        return True, ""

    def body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "front": self.front.to_dict(),
            "router": self.router.to_dict() if self.router else None,
            "specialist": self.specialist.to_dict() if self.specialist else None,
            "router_features": list(self.router_features),
            **(
                {
                    "harness": self.harness.to_dict() if self.harness else None,
                    "escalation": self.escalation.to_dict() if self.escalation else None,
                    "endpoint": self.endpoint.to_dict() if self.endpoint else None,
                }
                if self.full
                else {}
            ),
        }

    def digest(self) -> str:
        return canonical_hash(self.body())

    def locate(self, role: Role) -> str:
        component = self.component(role)
        if component is None:
            return ""
        return component.path or DEFAULT_PATHS[role]

    @classmethod
    def single(
        cls, artifact_digest: str, placement: str, base_model: str = ""
    ) -> SystemManifest:
        return cls(
            front=ComponentRef(
                role=Role.FRONT,
                artifact_digest=artifact_digest,
                placement=placement,
                base_model=base_model,
            )
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SystemManifest:
        def ref(role: Role, raw: Any) -> ComponentRef | None:
            if raw is None:
                return None
            if not isinstance(raw, dict):
                raise SystemManifestError(f"the {role.value} component is not an object")
            try:
                return ComponentRef(
                    role=Role(str(raw["role"])),
                    artifact_digest=str(raw["artifact_digest"]),
                    placement=str(raw["placement"]),
                    path=str(raw.get("path", "")),
                    base_model=str(raw.get("base_model", "")),
                )
            except (KeyError, ValueError) as exc:
                raise SystemManifestError(
                    f"the {role.value} component is malformed: {exc}"
                ) from exc

        try:
            front = ref(Role.FRONT, payload["front"])
            if front is None:
                raise SystemManifestError("a system manifest must declare a front component")
            harness = payload.get("harness")
            escalation = payload.get("escalation")
            endpoint = payload.get("endpoint")
            return cls(
                front=front,
                router=ref(Role.ROUTER, payload.get("router")),
                specialist=ref(Role.SPECIALIST, payload.get("specialist")),
                router_features=tuple(payload.get("router_features") or ()),
                schema_version=int(payload.get("schema_version", 1)),
                harness=HarnessRef(
                    package_digest=str(harness["package_digest"]),
                    runtime=str(harness["runtime"]),
                    path=str(harness.get("path", "harness")),
                )
                if isinstance(harness, dict)
                else None,
                escalation=EscalationRef(
                    model=str(escalation["model"]), revision=str(escalation["revision"])
                )
                if isinstance(escalation, dict)
                else None,
                endpoint=EndpointRef(worker=str(endpoint["worker"]), name=str(endpoint["name"]))
                if isinstance(endpoint, dict)
                else None,
            )
        except (KeyError, TypeError) as exc:
            raise SystemManifestError(f"system manifest is malformed: {exc}") from exc
