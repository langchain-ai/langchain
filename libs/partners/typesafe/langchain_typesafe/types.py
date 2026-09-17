"""Question and response types for the TypeSafe integration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal, TypeAlias

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, JsonValue

_QuestionContent: TypeAlias = str | dict[str, JsonValue] | list[JsonValue]
_StateValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | BaseMessage
    | Sequence["_StateValue"]
    | dict[str, "_StateValue"]
    | None
)

State: TypeAlias = str | BaseMessage | Sequence[_StateValue] | dict[str, _StateValue]
"""Root state accepted by `TypeSafeClassifier`.

TypeSafe natively accepts a string, JSON object, or JSON array. LangChain
`BaseMessage` objects and message sequences can appear at the root or at any depth
inside objects and arrays. The integration serializes messages as role/content JSON
while preserving surrounding JSON structure.
"""


class NoulCriteria(BaseModel):
    """Optional descriptions for the two outcomes of a `Noul` question.

    Criteria clarify what should count as yes and no when the instruction alone leaves
    room for interpretation. Both values accept any JSON-compatible content, so callers
    can provide a short description or structured examples.
    """

    model_config = ConfigDict(populate_by_name=True)

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
        from langchain_typesafe import Noul, TypeSafeClassifier

        classifier = TypeSafeClassifier(
            questions={
                "urgent": Noul(
                    instructions="Does this message require an urgent response?"
                )
            }
        )
        response = classifier.invoke("Production is down. Please help immediately.")
        urgency = response.nouls["urgent"].noul

        if urgency >= 0.8:
            page_on_call_engineer()
        ```
    """

    type: Literal["noul"] = "noul"
    """Wire discriminator for a binary TypeSafe question."""

    instructions: _QuestionContent
    """Complete yes/no judgment to make about the input state.

    Instructions may be text or structured JSON. Write the full question here even when
    the question ID used by `TypeSafeClassifier.questions` appears self-explanatory.
    """

    criteria: NoulCriteria | None = None
    """Optional descriptions that define what the yes and no outcomes mean."""


class Choice(BaseModel):
    """Select one label from a fixed set of alternatives.

    Use `Choice` for categorical decisions with no inherent ordering, such as routing a
    support request, detecting a document type, or selecting an intent. The answer
    includes the selected label, a probability for every supplied label, and confidence
    derived from the shape of that probability distribution.

    ??? example "Route a support request"

        ```python
        from langchain_typesafe import Choice, TypeSafeClassifier

        classifier = TypeSafeClassifier(
            questions={
                "department": Choice(
                    instructions="Which team should handle this request?",
                    criteria={
                        "billing": "Payment, invoice, or subscription issues.",
                        "technical": "Product bugs or integration failures.",
                        "sales": "Pricing or purchasing questions.",
                    },
                )
            }
        )
        response = classifier.invoke("Stripe fails whenever I connect my account.")
        department = response.choices["department"]

        if department.confidence >= 0.7:
            route_to(department.choice)
        else:
            route_to_human_triage()
        ```
    """

    type: Literal["choice"] = "choice"
    """Wire discriminator for a categorical TypeSafe question."""

    criteria: dict[str, JsonValue] = Field(min_length=1)
    """Candidate labels mapped to their descriptions.

    Descriptions may be text, structured JSON, or `None`. Include an `other` or
    `none_of_the_above` label when the supplied alternatives may not cover every input.
    """

    instructions: _QuestionContent
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
        from langchain_typesafe import Score, TypeSafeClassifier

        classifier = TypeSafeClassifier(
            questions={
                "frustration": Score(
                    instructions="How frustrated does the customer appear?",
                    criteria=[
                        "Calm and neutral.",
                        "Concerned but civil.",
                        "Very angry or using strong language.",
                    ],
                )
            }
        )
        response = classifier.invoke("This has failed three times. Fix it now.")
        frustration = response.scores["frustration"]

        print(frustration.score)  # May be fractional, for example 1.35.
        print(frustration.legend)  # The original zero-based rubric.
        print(frustration.probabilities)  # Probability for each rubric level.
        ```
    """

    type: Literal["score"] = "score"
    """Wire discriminator for an ordinal TypeSafe question."""

    criteria: list[JsonValue] = Field(min_length=2)
    """Two or more ordered descriptions for score levels starting at zero."""

    instructions: _QuestionContent
    """Complete ordinal judgment to make about the input state."""


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]
"""A discriminated union of question types accepted by `TypeSafeClassifier`."""


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
    or risk policy different from TypeSafe's default confidence calculation.
    """

    confidence: float = Field(ge=0.0, le=1.0)
    """Scalar certainty from `0` to `1`, derived from the probability distribution.

    Confidence describes how concentrated the distribution is; it is not the selected
    label's probability. Thresholds should be chosen according to the consequences of
    an incorrect automated decision.
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


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]
"""A discriminated union of answers returned by `TypeSafeClassifier`."""


class Usage(BaseModel):
    """Token usage reported for a TypeSafe classification request."""

    input_tokens: int | None = None
    """Number of input tokens processed, or `None` when not reported."""

    output_tokens: int | None = None
    """Number of output tokens produced, or `None` when not reported."""


class ClassificationResponse(BaseModel):
    """Typed answers and metadata returned from one TypeSafe request.

    Access every answer through `answers`, or use `nouls`, `choices`, and `scores` for
    views filtered by answer type. Each mapping preserves the question IDs supplied to
    `TypeSafeClassifier.questions`.
    """

    model: str
    """TypeSafe model that answered the request."""

    answers: dict[str, Answer]
    """All recognized answers keyed by their original question IDs."""

    usage: Usage = Field(default_factory=Usage)
    """Input and output token counts reported for the request."""

    request_id: str | None = None
    """TypeSafe request ID, useful when diagnosing a request with provider support."""

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


__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "ClassificationResponse",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "Question",
    "Score",
    "ScoreAnswer",
    "State",
    "Usage",
]
