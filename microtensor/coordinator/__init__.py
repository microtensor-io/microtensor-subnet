"""The coordinator: assigns measurement, aggregates reports, settles a round.

It performs no measurement itself. No model loads here, no miner artifact is
fetched, and no jail is installed. It reads the chain, serves JSON, stores
rows, and computes a settlement from numbers other people produced.

If a change proposes that the coordinator fetch or execute an artifact, that
change is turning it into a worker.
"""

from microtensor.coordinator.assign import System, Worker, assign, by_worker
from microtensor.coordinator.collect import Reconciled, ReportRejected, intake, reconcile
from microtensor.coordinator.report import CostBlock, QualityBlock, Report
from microtensor.coordinator.reputation import Standing
from microtensor.coordinator.settle import Entry, Settlement, merkle_root
from microtensor.coordinator.store import CoordinatorStore

__all__ = [
    "CoordinatorStore",
    "CostBlock",
    "Entry",
    "QualityBlock",
    "Reconciled",
    "Report",
    "ReportRejected",
    "Settlement",
    "Standing",
    "System",
    "Worker",
    "assign",
    "by_worker",
    "intake",
    "merkle_root",
    "reconcile",
]
