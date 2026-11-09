"""Question and response types shared by decision models."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
    with_config,
)
from typing_extensions import Self, TypedDict

from langchain_core.messages import BaseMessage, UsageMetadata

logger = logging.getLogger(__name__)

Content: TypeAlias = str | dict[str, JsonValue] | list[JsonValue]
"""Text, a JSON object, or a JSON array used in question instructions and criteria.

Providers that only accept text send JSON content as JSON text.
"""

_StateValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | BaseMessage
    | Sequence["_StateValue"]
    | Mapping[str, "_StateValue"]
    | None
)

State: TypeAlias = str | BaseMessage | Sequence[_StateValue] | Mapping[str, _StateValue]
"""State accepted by decision models: text, messages, or JSON-compatible data.

LangChain messages may appear at the root or nested inside JSON objects and arrays.
LangChain standard media blocks (for example `{"type": "image", "base64": ...,
"mime_type": ...}`) may appear anywhere in the state. Each provider documents which
media it supports and how it sends state its API cannot accept natively.
"""


@with_config(ConfigDict(extra="forbid"))
class NoulCriteria(TypedDict, total=False):
    """Optional descriptions for the two outcomes of a `Noul` question."""

    true: Content | None
    """Description of the yes outcome, or `None` when no clarification is needed."""

    false: Content | None
    """Description of the no outcome, or `None` when no clarification is needed."""


class Noul(BaseModel):
    """Ask a yes/no question and receive the probability that the answer is yes.

    A value near `1` favors yes, a value near `0` favors no, and a value near `0.5`
    indicates uncertainty.

    ??? example "Detect an urgent support request"

        ```python
        from langchain_core.decisions import Noul

        response = classifier.invoke(
            {
                "state": "Production is down. Please help immediately.",
                "questions": {
                    "urgent": Noul(
                        instructions="Does this message require an urgent response?"
                    )
                },
            }
        )
        if response.nouls["urgent"].noul >= 0.8:
            page_on_call_engineer()
        ```
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    """Discriminator for a yes/no question."""

    instructions: Content | None = None
    """Complete yes/no judgment to make about the state.

    Optional when `criteria` describes at least one outcome.
    """

    criteria: NoulCriteria | None = None
    """Optional descriptions that define what the yes and no outcomes mean."""

    @model_validator(mode="after")
    def _require_instructions_or_criteria(self) -> Self:
        if self.instructions is None and not any(
            value is not None for value in (self.criteria or {}).values()
        ):
            msg = "A Noul question requires `instructions` or a described outcome."
            raise ValueError(msg)
        return self


class Choice(BaseModel):
    """Select one label from a fixed, unordered set of alternatives.

    Include a fallback label such as `other` when the alternatives may not cover every
    input.

    ??? example "Route a support request"

        ```python
        from langchain_core.decisions import Choice

        response = classifier.invoke(
            {
                "state": "Stripe fails whenever I connect my account.",
                "questions": {
                    "department": Choice(
                        instructions="Which team should handle this request?",
                        criteria={
                            "billing": "Payment, invoice, or subscription issues.",
                            "technical": "Product bugs or integration failures.",
                        },
                    )
                },
            }
        )
        print(response.choices["department"].choice)
        ```
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    """Discriminator for a categorical question."""

    instructions: Content | None = None
    """Complete categorical judgment to make about the state."""

    criteria: dict[str, Content | None] = Field(min_length=1)
    """Candidate labels mapped to optional descriptions of when each applies."""


class Score(BaseModel):
    """Evaluate state against an ordered rubric.

    Criteria are numbered from zero in their supplied order. The returned score is the
    probability-weighted average of those levels, so it may fall between them.

    ??? example "Score customer frustration"

        ```python
        from langchain_core.decisions import Score

        response = classifier.invoke(
            {
                "state": "This has failed three times. Fix it now.",
                "questions": {
                    "frustration": Score(
                        instructions="How frustrated does the customer appear?",
                        criteria=["Calm.", "Concerned but civil.", "Very angry."],
                    )
                },
            }
        )
        print(response.scores["frustration"].score)  # For example, 1.35.
        ```
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    """Discriminator for an ordinal question."""

    instructions: Content | None = None
    """Complete ordinal judgment to make about the state."""

    criteria: list[Content] = Field(min_length=1)
    """Ordered descriptions of the score levels, starting at zero."""


class Box(BaseModel):
    """Locate what the instructions describe in an image in the state."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["box"] = "box"
    """Discriminator for a localization question."""

    instructions: Content | None = None
    """What to locate in the image."""


Question = Annotated[Noul | Choice | Score | Box, Field(discriminator="type")]
"""A discriminated union of question types accepted by decision models."""


class ClassifierRequest(TypedDict):
    """Complete input for one decision model invocation.

    Keeping the state and questions together in the `Runnable` input ensures both
    participate in composition, batching, and tracing.
    """

    state: State
    """Text, structured JSON, or LangChain messages to classify."""

    questions: Mapping[str, Question | Mapping[str, Any]]
    """Non-empty mapping of answer IDs to questions.

    Questions may be `Noul`, `Choice`, `Score`, or `Box` objects, or plain dicts with
    the same fields.
    """


class NoulAnswer(BaseModel):
    """Probability that a `Noul` question's answer is yes."""

    type: Literal["noul"]
    """Discriminator identifying a yes/no answer."""

    noul: float = Field(ge=0.0, le=1.0)
    """Probability of yes in the inclusive range from `0` to `1`."""


class ChoiceAnswer(BaseModel):
    """Selected `Choice` label with its probability distribution and confidence."""

    type: Literal["choice"]
    """Discriminator identifying a categorical answer."""

    choice: str
    """Label selected from the options supplied in `Choice.criteria`."""

    probabilities: dict[str, float]
    """Probability assigned to each candidate label, keyed by label."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Certainty from `0` to `1` in the selected label.

    Each provider documents how it derives confidence. Choose thresholds according to
    the consequences of an incorrect automated decision.
    """


class ScoreAnswer(BaseModel):
    """Expected `Score` value with its rubric, distribution, and confidence."""

    type: Literal["score"]
    """Discriminator identifying an ordinal answer."""

    score: float
    """Expected position on the ordered rubric, which may be fractional."""

    legend: dict[int, JsonValue]
    """Rubric descriptions keyed by their zero-based levels."""

    probabilities: dict[int, float]
    """Probability of each zero-based rubric level."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Certainty from `0` to `1` in the score."""


class BoxCoordinates(BaseModel):
    """Edges of a box as fractions of the image height and width, from `0` to `1`."""

    top: float
    """Top edge, as a fraction of the image height from the top."""

    bottom: float
    """Bottom edge, as a fraction of the image height from the top."""

    left: float
    """Left edge, as a fraction of the image width from the left."""

    right: float
    """Right edge, as a fraction of the image width from the left."""


class BoxAnswer(BaseModel):
    """Location of what a `Box` question describes."""

    type: Literal["box"]
    """Discriminator identifying a localization answer."""

    box: BoxCoordinates
    """Edges of the located box."""

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    """Certainty from `0` to `1`, or `None` when the provider does not report it."""


Answer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer | BoxAnswer,
    Field(discriminator="type"),
]
"""A discriminated union of answers returned by decision models."""

_ANSWER_TYPES = frozenset({"noul", "choice", "score", "box"})


class ClassifierResponse(BaseModel):
    """Typed answers and metadata returned from one classification request.

    Access every answer through `answers`, or use `nouls`, `choices`, `scores`, and
    `boxes` for views filtered by answer type. Each mapping preserves the question IDs
    supplied in `ClassifierRequest.questions`.

    Answers of a type this version does not recognize are dropped with a warning, so
    new provider answer types do not fail an otherwise successful request.
    """

    model_config = ConfigDict(extra="ignore")

    model: str
    """Model that answered the request."""

    answers: dict[str, Answer]
    """All recognized answers keyed by their question IDs."""

    usage: UsageMetadata | None = None
    """Token usage reported for the request, or `None` when not reported.

    Providers may supply only `input_tokens` and `output_tokens`. A missing count is
    treated as `0`, and `total_tokens` defaults to their sum.
    """

    request_id: str | None = None
    """Provider request ID, or `None` when not reported."""

    @field_validator("usage", mode="before")
    @classmethod
    def _complete_usage(cls, usage: Any) -> Any:
        if not isinstance(usage, Mapping):
            return usage
        input_tokens = usage.get("input_tokens") or 0
        output_tokens = usage.get("output_tokens") or 0
        return {
            **usage,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens") or input_tokens + output_tokens,
        }

    @model_validator(mode="before")
    @classmethod
    def _drop_unknown_answers(cls, data: Any) -> Any:
        if not isinstance(data, Mapping) or not isinstance(
            answers := data.get("answers"), Mapping
        ):
            return data
        known = {}
        for name, answer in answers.items():
            answer_type = (
                answer.get("type")
                if isinstance(answer, Mapping)
                else getattr(answer, "type", None)
            )
            if answer_type in _ANSWER_TYPES:
                known[name] = answer
            else:
                logger.warning(
                    "Skipping answer %r with unsupported type %r.", name, answer_type
                )
        return {**data, "answers": known}

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        """Return yes/no answers keyed by their question IDs."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, NoulAnswer)
        }

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """Return categorical answers keyed by their question IDs."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ChoiceAnswer)
        }

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """Return ordinal answers keyed by their question IDs."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ScoreAnswer)
        }

    @property
    def boxes(self) -> dict[str, BoxAnswer]:
        """Return localization answers keyed by their question IDs."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, BoxAnswer)
        }


__all__ = [
    "Answer",
    "Box",
    "BoxAnswer",
    "BoxCoordinates",
    "Choice",
    "ChoiceAnswer",
    "ClassifierRequest",
    "ClassifierResponse",
    "Content",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "Question",
    "Score",
    "ScoreAnswer",
    "State",
]
