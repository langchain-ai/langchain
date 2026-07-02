"""Data models for tool security scanning results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal


class Severity(IntEnum):
    """Finding severity levels, ordered from lowest to highest."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


SinkChannel = Literal["stdout", "return", "logging", "network", "hardcoded_secret"]


@dataclass(frozen=True)
class SecurityFinding:
    """A single security finding from tool analysis."""

    rule_id: str
    severity: Severity
    message: str
    file: str
    line: int
    channel: SinkChannel
    tool_name: str | None = None
    column: int | None = None
    attack_flow: tuple[str, ...] = ()
    cross_modal: bool = False


@dataclass
class ScanResult:
    """Aggregated results from scanning one or more tools."""

    findings: list[SecurityFinding] = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        """Whether any findings were reported."""
        return bool(self.findings)

    def merge(self, other: ScanResult) -> ScanResult:
        """Merge another scan result into this one."""
        self.findings.extend(other.findings)
        return self

    def findings_at_or_above(self, min_severity: Severity) -> list[SecurityFinding]:
        """Return findings at or above the given severity."""
        return [f for f in self.findings if f.severity >= min_severity]

    def raise_on_findings(self, *, min_severity: Severity = Severity.HIGH) -> None:
        """Raise ``ValueError`` if findings meet the severity threshold.

        Args:
            min_severity: Minimum severity that triggers an error.

        Raises:
            ValueError: If one or more findings meet the threshold.
        """
        matched = self.findings_at_or_above(min_severity)
        if not matched:
            return
        lines = "\n".join(
            f"  [{f.severity.name}] {f.rule_id} {f.file}:{f.line} — {f.message}"
            for f in matched
        )
        msg = f"Tool security scan found {len(matched)} issue(s):\n{lines}"
        raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the scan result to a JSON-compatible dictionary."""
        return {
            "findings": [
                {
                    "rule_id": f.rule_id,
                    "severity": f.severity.name,
                    "message": f.message,
                    "file": f.file,
                    "line": f.line,
                    "column": f.column,
                    "channel": f.channel,
                    "tool_name": f.tool_name,
                    "attack_flow": list(f.attack_flow),
                    "cross_modal": f.cross_modal,
                }
                for f in self.findings
            ],
            "count": len(self.findings),
        }

    def format(self) -> str:
        """Return a human-readable summary of findings."""
        if not self.findings:
            return "No tool security findings."
        lines = [
            f"[{f.severity.name}] {f.rule_id} {f.file}:{f.line} — {f.message}"
            for f in self.findings
        ]
        return "\n".join(lines)
