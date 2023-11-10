"""Base class for token usage reporters."""

import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class TokenUsageReport(BaseModel):
    """Token usage report data class."""

    model_config = ConfigDict(protected_namespaces=())

    timestamp: datetime.datetime = Field(default=..., description="The timestamp of the llm run.")
    prompt_tokens: int | None = Field(default=None, description="Number of prompt tokens consumed.")
    completion_tokens: int | None = Field(
        default=None, description="Number of completion tokens consumed."
    )
    total_tokens: int | None = Field(default=None, description="Number of total tokens consumed.")
    total_cost: float | None = Field(default=None, description="Estimated total cost.")
    first_token_time: float | None = Field(
        default=None,
        description="Elapsed time in seconds until the first token was emitted.",
    )
    completion_time: float | None = Field(
        default=None, description="Elapsed time in seconds of the completion."
    )
    model_name: str | None = Field(default=None, description="Name and variant of the model used.")
    caller_id: str | None = Field(
        default=None,
        description="Identifier of the caller (eg. API key fraction, org name, etc.)",
    )


class TokenUsageReporter(Protocol):
    """A generic interface to report LLM token usage."""

    def send_report(self, report: TokenUsageReport) -> None:
        """Reports token usage statistics of a single LLM run.

        Args:
            report (TokenUsageReport): The report to be sent.
        """
        ...
