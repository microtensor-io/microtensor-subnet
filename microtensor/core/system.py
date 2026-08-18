from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

from microtensor.core.constants import ALLOWED_ROUTER_FEATURES, HOST_PROFILE
from microtensor.core.hashing import canonical_hash
from microtensor.core.protocol import Role


class SystemManifestError(ValueError):
    pass


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

    def __post_init__(self) -> None:
        if self.front.role is not Role.FRONT:
            raise SystemManifestError("the front slot must carry a front component")
        if self.router is not None and self.router.role is not Role.ROUTER:
            raise SystemManifestError("the router slot must carry a router component")
        if self.specialist is not None and self.specialist.role is not Role.SPECIALIST:
            raise SystemManifestError("the specialist slot must carry a specialist component")

        if (self.router is None) != (self.specialist is None):
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
        return self.router is None and self.specialist is None

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
            return cls(
                front=front,
                router=ref(Role.ROUTER, payload.get("router")),
                specialist=ref(Role.SPECIALIST, payload.get("specialist")),
                router_features=tuple(payload.get("router_features") or ()),
                schema_version=int(payload.get("schema_version", 1)),
            )
        except (KeyError, TypeError) as exc:
            raise SystemManifestError(f"system manifest is malformed: {exc}") from exc
