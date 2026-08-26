from dataclasses import dataclass, asdict
from typing import Any, Optional


@dataclass
class ToolResult:
    ok: bool
    data: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def success(data: Any) -> dict:
    return ToolResult(ok=True, data=data).to_dict()


def failure(error: str) -> dict:
    return ToolResult(ok=False, error=error).to_dict()
