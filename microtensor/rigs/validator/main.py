from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import time
from typing import Any

from microtensor.rigs.protocol.session import release_message
from microtensor.rigs.validator import __version__
from microtensor.rigs.validator.config import Settings
from microtensor.rigs.validator.config import settings as load_settings
from microtensor.rigs.validator.identity import Identity, IdentityError, load_keypair
from microtensor.rigs.validator.pool import PoolClient, ServerError
from microtensor.rigs.validator.schedule.deep import DeepPass
from microtensor.rigs.validator.schedule.express import ExpressLane
from microtensor.rigs.validator.scoring.demand_weight import DemandWeights
from microtensor.rigs.validator.scoring.reliability import UptimeLedger
from microtensor.rigs.validator.scoring.weights import Chain, compute_scores
from microtensor.rigs.validator.session.client import AgentClient

log = logging.getLogger("validator")

ACTIVATION_POLL_SECONDS = 60.0


class Validator:
    def __init__(self, settings: Settings, keypair: Any = None, gate: Any = None) -> None:
        self.settings = settings
        self.gate = gate
        self.identity = Identity(keypair if keypair is not None else load_keypair(settings))
        self.pool = PoolClient(settings, self.identity)
        self.agents = AgentClient(self.identity, ssh_timeout=float(settings.ssh_timeout))
        self.chain = Chain(settings.chain_endpoint, settings.netuid, settings.mechanism_id)
        self.ledger = UptimeLedger(settings.path("reliability.json"))
        self.demand = DemandWeights(settings.path("demand.json"))
        self.express = ExpressLane(settings, self.pool, self.agents, self.ledger)
        self.deep = DeepPass(settings, self.pool, self.agents, self.ledger, self.express)
        self.active = False
        self.last_weights = 0.0

    async def close(self) -> None:
        await self.agents.close()
        await self.pool.close()
        self.chain.close()

    async def ensure_registered(self) -> bool:
        try:
            me = await self.pool.me()
        except ServerError as exc:
            if exc.status not in (401, 403, 404):
                raise
            me = await self.pool.register(self.settings.label or self.identity.hotkey[:12])
        self.active = str(me.get("state", "")) == "active"
        if not self.active:
            log.warning(
                "validator %s is %s; waiting for the operator",
                self.identity.hotkey,
                me.get("state"),
            )
        return self.active

    async def wait_active(self) -> None:
        while True:
            try:
                if await self.ensure_registered():
                    log.info("validator %s is active", self.identity.hotkey)
                    return
            except ServerError as exc:
                log.warning("registration check failed: %s", exc.detail)
            await asyncio.sleep(ACTIVATION_POLL_SECONDS)

    async def scores_by_hotkey(self) -> tuple[dict[str, float], str, set[str]]:
        roster = await self.pool.rigs()
        rigs = [rig for rig in roster.get("rigs") or [] if isinstance(rig, dict)]
        try:
            server_scores = await self.pool.scores()
        except ServerError as exc:
            log.warning(
                "scores unavailable from the server, using the local formula: %s", exc.detail
            )
            server_scores = None
        scores, origin = compute_scores(
            server_scores,
            rigs,
            self.demand,
            self.ledger,
            min_driver_version=self.settings.min_driver_version,
            driver_cutoff_passed=self.settings.driver_cutoff_passed(),
        )
        active = {
            str(rig.get("hotkey", "")) for rig in rigs if rig.get("online") and rig.get("hotkey")
        }
        return scores, origin, active

    async def set_weights_once(self, dry_run: bool = False) -> dict[str, Any]:
        scores, origin, active = await self.scores_by_hotkey()
        if not scores:
            return {
                "submitted": False,
                "dry_run": dry_run,
                "origin": origin,
                "reason": "no scores yet",
            }
        outcome = await asyncio.to_thread(
            self.chain.submit,
            self.identity.keypair,
            scores,
            active,
            self.settings.reserve_uid,
            self.settings.held_share,
            dry_run,
        )
        outcome["origin"] = origin
        if self.settings.reserve_uid is None and self.settings.held_share > 0:
            outcome["note"] = (
                "held share not routed: CV_RESERVE_UID is unset, shares renormalised over miners"
            )
        return outcome

    async def weights_loop(self) -> None:
        if not self.settings.set_weights:
            log.info("weights are off (CV_SET_WEIGHTS=0)")
            return
        while True:
            if time.time() - self.last_weights >= self.settings.weights_seconds:
                try:
                    outcome = await self.set_weights_once()
                    log.info("set_weights %s", json.dumps(outcome, default=str))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.exception("set_weights failed: %s", exc)
                self.last_weights = time.time()
            await asyncio.sleep(30.0)

    async def run(self) -> int:
        await self.wait_active()
        log.info(
            "running express every %ss, deep pass poll every 60s, weights every %ss (%s)",
            self.settings.express_seconds,
            self.settings.weights_seconds,
            "on" if self.settings.set_weights else "off",
        )
        try:
            await asyncio.gather(
                self.express.loop(self.gate), self.deep.loop(self.gate), self.weights_loop()
            )
        finally:
            self.ledger.save()
            await self.close()
        return 0


async def _with_pool(settings: Settings, action: Any) -> int:
    identity = Identity(load_keypair(settings))
    async with PoolClient(settings, identity) as pool:
        answer = await action(pool, identity)
    print(json.dumps(answer, indent=2, default=str))
    return 0


async def _show(settings: Settings, what: str) -> int:
    async def action(pool: PoolClient, identity: Identity) -> Any:
        if what == "scores":
            return await pool.scores()
        return await pool.rigs(due=what == "due")

    return await _with_pool(settings, action)


async def _sign_release(settings: Settings, digest: str) -> int:
    async def action(pool: PoolClient, identity: Identity) -> Any:
        timestamp = int(time.time())
        signature = identity.sign(release_message(digest, timestamp))
        return await pool.release(digest, timestamp, signature)

    return await _with_pool(settings, action)


async def _register(settings: Settings) -> int:
    validator = Validator(settings)
    try:
        active = await validator.ensure_registered()
    finally:
        await validator.close()
    print("active" if active else "registered, awaiting operator activation")
    return 0


async def _weights(settings: Settings, dry_run: bool) -> int:
    validator = Validator(settings)
    try:
        outcome = await validator.set_weights_once(dry_run=dry_run)
    finally:
        await validator.close()
    print(json.dumps(outcome, indent=2, default=str))
    return 0 if outcome.get("submitted") or dry_run else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compute-validator")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run")
    sub.add_parser("register")
    sub.add_parser("hotkey")
    sub.add_parser("version")
    show = sub.add_parser("show")
    show.add_argument("what", choices=["scores", "rigs", "due"])
    release = sub.add_parser("sign-release")
    release.add_argument("digest")
    weights = sub.add_parser("weights")
    weights.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = load_settings()
    if args.command == "version":
        print(__version__)
        return 0
    try:
        if args.command == "hotkey":
            print(Identity(load_keypair(settings)).hotkey)
            return 0
        if args.command == "register":
            return asyncio.run(_register(settings))
        if args.command == "show":
            return asyncio.run(_show(settings, args.what))
        if args.command == "sign-release":
            return asyncio.run(_sign_release(settings, args.digest))
        if args.command == "weights":
            return asyncio.run(_weights(settings, args.dry_run))
        if args.command == "run":
            with contextlib.suppress(KeyboardInterrupt):
                return asyncio.run(Validator(settings).run())
            return 0
    except IdentityError as exc:
        print(f"identity: {exc}", file=sys.stderr)
        return 2
    except ServerError as exc:
        print(f"server: {exc}", file=sys.stderr)
        return 3
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
