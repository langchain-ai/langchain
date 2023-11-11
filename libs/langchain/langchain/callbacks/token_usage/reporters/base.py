"""Base class for token usage reporters."""

import datetime
from typing import Optional, Protocol

from langchain.pydantic_v1 import BaseModel, Field


class TokenUsageReport(BaseModel):
    """Token usage report data class."""

    timestamp: datetime.datetime = Field(
        default=..., description="The timestamp of the llm run."
    )
    prompt_tokens: Optional[int] = Field(
        default=None, description="Number of prompt tokens consumed."
    )
    completion_tokens: Optional[int] = Field(
        default=None, description="Number of completion tokens consumed."
    )
    total_tokens: Optional[int] = Field(
        default=None, description="Number of total tokens consumed."
    )
    total_cost: Optional[float] = Field(
        default=None, description="Estimated total cost."
    )
    first_token_time: Optional[float] = Field(
        default=None,
        description="Elapsed time in seconds until the first token was emitted.",
    )
    completion_time: Optional[float] = Field(
        default=None, description="Elapsed time in seconds of the completion."
    )
    model_name: Optional[str] = Field(
        default=None, description="Name and variant of the model used."
    )
    caller_id: Optional[str] = Field(
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
