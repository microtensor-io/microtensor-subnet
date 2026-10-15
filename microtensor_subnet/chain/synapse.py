from typing import Any

_SYSTEM_TASK: Any = None


def system_task() -> Any:
    global _SYSTEM_TASK
    if _SYSTEM_TASK is None:
        import bittensor as bt
        from pydantic import Field

        class SystemTask(bt.Synapse):  # type: ignore[misc]
            system: str = ""
            round_index: int = 0
            task_ref: str = ""
            prompt: str = ""
            inputs: dict[str, Any] = Field(default_factory=dict)
            trace: dict[str, Any] | None = None
            failure: str = ""

            def deserialize(self) -> dict[str, Any] | None:
                return self.trace

        _SYSTEM_TASK = SystemTask
    return _SYSTEM_TASK
