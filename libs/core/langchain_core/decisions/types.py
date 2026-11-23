"""Question and response types shared by decision models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, UsageMetadata

Content: TypeAlias = str | dict[str, JsonValue] | list[JsonValue]
"""Text, a JSON object, or a JSON array used in question instructions.

As in `State`, LangChain standard media blocks may appear anywhere in instructions and
criteria. Providers send them as media, or raise an error if their API does not accept
that media in questions.
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
"""Root state accepted by decision models.

State is a string, JSON object, or JSON array. LangChain `BaseMessage` objects and
message sequences can appear at the root or at any depth inside objects and arrays.
LangChain standard media blocks (for example `{"type": "image", "base64": ...,
"mime_type": ...}`) may appear anywhere in the state. Each provider documents which
media it supports and how it sends state its API cannot accept natively.
"""


class NoulCriteria(BaseModel):
    """Optional descriptions for the two outcomes of a `Noul` question.

    Criteria clarify what should count as yes and no when the instruction alone leaves
    room for interpretation. Both values accept any JSON-compatible content, so callers
    can provide a short description or structured examples.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    true: JsonValue = None
    """Description of the yes outcome, or `None` when no clarification is needed."""

    false: JsonValue = None
    """Description of the no outcome, or `None` when no clarification is needed."""


class Noul(BaseModel):
    """Ask a binary question and receive the probability that its answer is yes.

    Use `Noul` when the probability itself is useful to application code, such as
    deciding whether a message reports a bug or requests a refund. A value near `1`
    indicates strong support for yes, a value near `0` indicates strong support for no,
    and a value near `0.5` indicates uncertainty. Noul answers do not include a separate
    confidence value.

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
        urgency = response.nouls["urgent"].noul

        if urgency >= 0.8:
            page_on_call_engineer()
        ```
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    """Wire discriminator for a binary question."""

    instructions: Content
    """Complete yes/no judgment to make about the input state.

    Instructions may be text or structured JSON. Write the full question here even when
    its ID in `ClassifierRequest.questions` appears self-explanatory.
    """

    criteria: NoulCriteria | None = None
    """Optional descriptions that define what the yes and no outcomes mean."""


class Choice(BaseModel):
    """Select one label from a fixed set of alternatives.

    Use `Choice` for categorical decisions with no inherent ordering, such as routing a
    support request, detecting a document type, or selecting an intent. The answer
    includes the selected label, a probability for every supplied label, and a
    confidence value.

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
                            "sales": "Pricing or purchasing questions.",
                        },
                    )
                },
            }
        )
        department = response.choices["department"]

        if department.confidence >= 0.7:
            route_to(department.choice)
        else:
            route_to_human_triage()
        ```
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    """Wire discriminator for a categorical question."""

    criteria: dict[str, JsonValue] = Field(min_length=1)
    """Candidate labels mapped to their descriptions.

    Descriptions may be text, structured JSON, or `None`. Include an `other` or
    `none_of_the_above` label when the supplied alternatives may not cover every input.
    """

    instructions: Content
    """Complete categorical judgment to make about the input state."""


class Score(BaseModel):
    """Evaluate state against an ordered rubric.

    Use `Score` when the answer lies on a spectrum whose levels can be described, such
    as severity, urgency, or customer frustration. Criteria are numbered from zero in
    their supplied order. The returned score is an expected value and may fall between
    integer levels; the full probability distribution remains available for custom
    decision logic.

    ??? example "Score customer frustration"

        ```python
        from langchain_core.decisions import Score

        response = classifier.invoke(
            {
                "state": "This has failed three times. Fix it now.",
                "questions": {
                    "frustration": Score(
                        instructions="How frustrated does the customer appear?",
                        criteria=[
                            "Calm and neutral.",
                            "Concerned but civil.",
                            "Very angry or using strong language.",
                        ],
                    )
                },
            }
        )
        frustration = response.scores["frustration"]

        print(frustration.score)  # May be fractional, for example 1.35.
        print(frustration.legend)  # The original zero-based rubric.
        print(frustration.probabilities)  # Probability for each rubric level.
        ```
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    """Wire discriminator for an ordinal question."""

    criteria: list[JsonValue] = Field(min_length=2)
    """Two or more ordered descriptions for score levels starting at zero."""

    instructions: Content
    """Complete ordinal judgment to make about the input state."""


class Box(BaseModel):
    """Locate what the instructions describe in an image in the state.

    The answer gives the edges of a box as fractions of the image height and width.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["box"] = "box"
    """Wire discriminator for a localization question."""

    instructions: Content
    """What to locate in the image."""


Question = Annotated[Noul | Choice | Score | Box, Field(discriminator="type")]
"""A discriminated union of question types accepted by decision models."""


class ClassifierRequest(TypedDict):
    """Complete input for one decision model invocation.

    Keeping the state and questions in the Runnable input ensures both values
    participate in composition, batching, and tracing.
    """

    state: State
    """Text, structured JSON, or LangChain messages to classify."""

    questions: Mapping[str, Question]
    """Non-empty mapping of answer IDs to typed classification questions."""


class NoulAnswer(BaseModel):
    """Probability that a `Noul` question's answer is yes."""

    type: Literal["noul"]
    """Wire discriminator identifying a binary answer."""

    noul: float = Field(ge=0.0, le=1.0)
    """Probability of yes in the inclusive range from `0` to `1`."""


class ChoiceAnswer(BaseModel):
    """Selected `Choice` label with its probability distribution and confidence."""

    type: Literal["choice"]
    """Wire discriminator identifying a categorical answer."""

    choice: str
    """Label selected from the options supplied in `Choice.criteria`."""

    probabilities: dict[str, float]
    """Probability assigned to each candidate label, keyed by label name.

    The complete distribution is retained so applications can use a confidence measure
    or risk policy different from the provider's.
    """

    confidence: float = Field(ge=0.0, le=1.0)
    """Scalar certainty from `0` to `1` in the selected label.

    Each provider documents how it derives confidence. Thresholds should be chosen
    according to the consequences of an incorrect automated decision.
    """


class ScoreAnswer(BaseModel):
    """Expected `Score` value with its rubric, distribution, and confidence."""

    type: Literal["score"]
    """Wire discriminator identifying an ordinal answer."""

    score: float
    """Expected position on the ordered rubric.

    This value may be fractional because it summarizes the probability distribution
    over integer rubric levels rather than selecting exactly one level.
    """

    legend: dict[int, JsonValue]
    """Original rubric descriptions keyed by their zero-based integer levels."""

    probabilities: dict[int, float]
    """Probability distribution over the zero-based integer rubric levels."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Scalar certainty from `0` to `1`, derived from the level distribution."""


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
    """Wire discriminator identifying a localization answer."""

    box: BoxCoordinates
    """Edges of the located box."""

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    """Certainty from `0` to `1`, or `None` when the provider does not report it."""


Answer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer | BoxAnswer,
    Field(discriminator="type"),
]
"""A discriminated union of answers returned by decision models."""


class ClassifierResponse(BaseModel):
    """Typed answers and metadata returned from one classification request.

    Access every answer through `answers`, or use `nouls`, `choices`, `scores`, and
    `boxes` for views filtered by answer type. Each mapping preserves the question IDs
    supplied to `ClassifierRequest.questions`.
    """

    model: str
    """Model that answered the request."""

    answers: dict[str, Answer]
    """All answers keyed by their original question IDs."""

    usage: UsageMetadata | None = None
    """Token usage reported for the request, or `None` when not reported.

    Providers may supply only `input_tokens` and `output_tokens`. A missing count is
    treated as `0`, and `total_tokens` defaults to their sum.
    """

    request_id: str | None = None
    """Provider request ID, useful when diagnosing a request with provider support."""

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

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        """Return binary answers keyed by their original question IDs."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, NoulAnswer)
        }

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """Return categorical answers keyed by their original question IDs."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ChoiceAnswer)
        }

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """Return ordinal answers keyed by their original question IDs."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ScoreAnswer)
        }

    @property
    def boxes(self) -> dict[str, BoxAnswer]:
        """Return localization answers keyed by their original question IDs."""
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
