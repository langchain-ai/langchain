"""Question and response types shared by decision models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

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

DecisionInput: TypeAlias = (
    str | BaseMessage | Sequence[_StateValue] | dict[str, _StateValue]
)
"""Input accepted by decision models: text, messages, or JSON-compatible state.

Messages may appear at the root or nested inside JSON objects and arrays. Each
provider documents how it sends input its API cannot accept natively.
"""


class Predicate(BaseModel):
    """Estimate the probability that a condition is true."""

    type: Literal["predicate"] = "predicate"
    """Wire discriminator for a predicate question."""

    instructions: str = Field(min_length=1)
    """Condition to evaluate against the input."""


class Choice(BaseModel):
    """Select one value from a fixed, unordered set of options."""

    type: Literal["choice"] = "choice"
    """Wire discriminator for a choice question."""

    instructions: str = Field(min_length=1)
    """Judgment to make about the input."""

    choices: dict[str | bool, str | None] = Field(min_length=1)
    """Candidate values mapped to optional descriptions of when each applies.

    Include a fallback such as `"other"` when the options may not cover every input.
    """


class Level(BaseModel):
    """One level of a `Score` rubric."""

    label: str = Field(min_length=1)
    """Short name for the level."""

    description: str | None = None
    """Criteria that distinguish this level from its neighbors."""


class Score(BaseModel):
    """Rate the input against ordered levels.

    Levels are indexed from `0` in the order supplied. The returned score is the
    probability-weighted average of those indices, so it may fall between levels.
    """

    type: Literal["score"] = "score"
    """Wire discriminator for a score question."""

    instructions: str = Field(min_length=1)
    """Judgment to make about the input."""

    levels: list[str | Level] = Field(min_length=2)
    """Two or more levels ordered from lowest to highest.

    A plain string is shorthand for a level with only a label.
    """

    def as_levels(self) -> list[Level]:
        """Return the levels with string shorthand expanded to `Level` objects.

        Returns:
            One `Level` per rubric level, in order.
        """
        return [
            Level(label=level) if isinstance(level, str) else level
            for level in self.levels
        ]


Question = Annotated[Predicate | Choice | Score, Field(discriminator="type")]
"""A discriminated union of question types accepted by a decision model."""


class DecisionRequest(TypedDict):
    """Complete input for one decision model invocation.

    Keeping the input and questions together in the Runnable input ensures both
    participate in composition, batching, and tracing.
    """

    input: DecisionInput
    """Text, messages, or structured state to evaluate."""

    questions: dict[str, Question]
    """Mapping of question names to questions. Names key the returned answers."""


class PredicateAnswer(BaseModel):
    """Probability that a `Predicate` condition is true."""

    type: Literal["predicate"]
    """Wire discriminator identifying a predicate answer."""

    probability: float = Field(ge=0.0, le=1.0)
    """Estimated probability, from `0` to `1`, that the condition is true."""


class ChoiceAnswer(BaseModel):
    """Selected `Choice` value with its probability distribution and confidence."""

    type: Literal["choice"]
    """Wire discriminator identifying a choice answer."""

    choice: str | bool
    """Value selected from `Choice.choices`."""

    probabilities: dict[str | bool, float]
    """Probability assigned to each candidate value."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Scalar certainty from `0` to `1`, derived from the distribution.

    Confidence describes how concentrated the distribution is; it is not the selected
    value's probability.
    """


class ScoreAnswer(BaseModel):
    """Expected `Score` value with its levels, distribution, and confidence."""

    type: Literal["score"]
    """Wire discriminator identifying a score answer."""

    score: float
    """Probability-weighted average of the zero-based level indices."""

    legend: dict[int, str]
    """Level labels keyed by their zero-based indices."""

    probabilities: dict[int, float]
    """Probability assigned to each zero-based level index."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Scalar certainty from `0` to `1`, derived from the distribution."""


class RefusalAnswer(BaseModel):
    """The model declined to answer a question."""

    model_config = ConfigDict(extra="allow")

    type: Literal["refusal"]
    """Wire discriminator identifying a refusal."""


Answer = Annotated[
    PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer,
    Field(discriminator="type"),
]
"""A discriminated union of answers returned by a decision model."""


class Usage(BaseModel):
    """Token usage reported for a decision request."""

    input_tokens: int | None = None
    """Number of input tokens processed, or `None` when not reported."""

    output_tokens: int | None = None
    """Number of output tokens produced, or `None` when not reported."""


class DecisionResponse(BaseModel):
    """Typed answers and metadata returned from one decision request.

    Access every answer through `answers`, or use `predicates`, `choices`, `scores`,
    and `refusals` for views filtered by answer type. Each mapping is keyed by the
    question names supplied in `DecisionRequest.questions`.
    """

    model: str
    """Model that answered the request."""

    answers: dict[str, Answer]
    """All recognized answers keyed by question name."""

    usage: Usage = Field(default_factory=Usage)
    """Token counts reported for the request."""

    request_id: str | None = None
    """Provider request ID, useful when diagnosing a request with provider support."""

    @property
    def predicates(self) -> dict[str, PredicateAnswer]:
        """Return predicate answers keyed by question name."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, PredicateAnswer)
        }

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """Return choice answers keyed by question name."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ChoiceAnswer)
        }

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """Return score answers keyed by question name."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ScoreAnswer)
        }

    @property
    def refusals(self) -> dict[str, RefusalAnswer]:
        """Return refusals keyed by question name."""
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, RefusalAnswer)
        }


__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "DecisionInput",
    "DecisionRequest",
    "DecisionResponse",
    "Level",
    "Predicate",
    "PredicateAnswer",
    "Question",
    "RefusalAnswer",
    "Score",
    "ScoreAnswer",
    "Usage",
]
