from __future__ import annotations

import contextlib
from typing import Any

import httpx

from microtensor.rigs.validator.config import Settings
from microtensor.rigs.validator.identity import Identity, canonical_json

VALIDATORS = "/v1/compute/validators"
POOL = "/v1/pool"
TIMEOUT_SECONDS = 60.0


class ServerError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class PoolClient:
    def __init__(
        self, settings: Settings, identity: Identity, timeout: float = TIMEOUT_SECONDS
    ) -> None:
        self._identity = identity
        self._http = httpx.AsyncClient(base_url=settings.server_url.rstrip("/"), timeout=timeout)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> PoolClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _call(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        signed: bool = True,
    ) -> Any:
        raw = b""
        headers: dict[str, str] = {"accept": "application/json"}
        if body is not None:
            raw = canonical_json(body)
            headers["content-type"] = "application/json"
        if signed:
            headers.update(self._identity.headers(method, path, raw))
        try:
            answer = await self._http.request(
                method,
                path,
                params=params,
                content=raw if body is not None else None,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise ServerError(0, f"{method} {path}: {exc}") from exc
        if answer.status_code >= 400:
            detail = answer.text[:500]
            with contextlib.suppress(Exception):
                detail = str(answer.json().get("detail", detail))
            raise ServerError(answer.status_code, detail)
        if not answer.content:
            return {}
        try:
            return answer.json()
        except ValueError as exc:
            raise ServerError(answer.status_code, "response is not JSON") from exc

    async def register(self, label: str) -> dict[str, Any]:
        return dict(await self._call("POST", f"{VALIDATORS}/register", {"label": label}))

    async def me(self) -> dict[str, Any]:
        return dict(await self._call("GET", f"{VALIDATORS}/me"))

    async def hardware(self) -> dict[str, Any]:
        return dict(await self._call("GET", f"{VALIDATORS}/hardware"))

    async def rigs(self, *, due: bool = False) -> dict[str, Any]:
        params = {"due": "true"} if due else None
        return dict(await self._call("GET", f"{VALIDATORS}/rigs", params=params))

    async def rig(self, rig_id: str) -> dict[str, Any]:
        return dict(await self._call("GET", f"{VALIDATORS}/rigs/{rig_id}"))

    async def command(self, rig_id: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(
            await self._call(
                "POST", f"{VALIDATORS}/rigs/{rig_id}/commands", {"kind": kind, "payload": payload}
            )
        )

    async def command_result(self, command_id: int) -> dict[str, Any]:
        return dict(await self._call("GET", f"{VALIDATORS}/commands/{int(command_id)}"))

    async def verification(self, rig_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(await self._call("POST", f"{VALIDATORS}/rigs/{rig_id}/verifications", payload))

    async def place_job(self, rig_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(await self._call("POST", f"{VALIDATORS}/rigs/{rig_id}/jobs", payload))

    async def finish_job(self, job_id: str, state: str, revenue_tao: float = 0.0) -> dict[str, Any]:
        return dict(
            await self._call(
                "POST",
                f"{VALIDATORS}/jobs/{job_id}/finish",
                {"state": state, "revenue_tao": max(0.0, float(revenue_tao))},
            )
        )

    async def scores(self) -> dict[str, Any]:
        return dict(await self._call("GET", f"{VALIDATORS}/scores"))

    async def release(self, digest: str, timestamp: int, signature: str) -> dict[str, Any]:
        return dict(
            await self._call(
                "POST",
                f"{VALIDATORS}/release",
                {"digest": digest, "timestamp": int(timestamp), "signature": signature},
            )
        )

    async def validators(self) -> list[str]:
        answer = await self._call("GET", f"{POOL}/validators", signed=False)
        return [str(item) for item in (answer.get("validators") or [])]

    async def agent_release(self) -> dict[str, Any]:
        return dict(await self._call("GET", f"{POOL}/agent-release", signed=False))
