"""Serializable execution records for frontend integration and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    duration_seconds: float = 0


@dataclass
class Attempt:
    number: int
    code: str
    success: bool
    stdout: str = ""
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)
    duration_seconds: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class AnalysisResult:
    success: bool
    question: str
    run_id: str
    provider: str
    model: str
    stdout: str = ""
    code: str = ""
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def pass_at_1(self) -> bool:
        return bool(self.attempts and self.attempts[0].success)

    @property
    def retry_count(self) -> int:
        return max(0, len(self.attempts) - 1)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "pass_at_1": self.pass_at_1, "retry_count": self.retry_count}
