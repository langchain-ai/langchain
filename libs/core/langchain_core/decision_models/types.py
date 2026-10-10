"""Typed questions and answers for decision models.

!!! warning
    This interface is in beta. Provider-native types remain separate from these
    canonical types; shared names do not imply identical provider capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self, TypeAliasType, TypedDict

from langchain_core.messages import BaseMessage

if TYPE_CHECKING:
    _StateValue: TypeAlias = (
        str
        | int
        | float
        | bool
        | BaseMessage
        | list["_StateValue"]
        | dict[str, "_StateValue"]
        | None
    )
else:
    # A named recursive alias lets Pydantic generate finite JSON schemas.
    _StateValue = TypeAliasType(
        "_StateValue",
        str
        | int
        | float
        | bool
        | BaseMessage
        | list["_StateValue"]
        | dict[str, "_StateValue"]
        | None,
    )
State: TypeAlias = str | BaseMessage | list[_StateValue] | dict[str, _StateValue]
"""Text, JSON-compatible state, or messages, including nested messages."""

_Probability: TypeAlias = Annotated[
    float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)
]
_Count: TypeAlias = Annotated[int, Field(strict=True, ge=0)]


class _DecisionData(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, revalidate_instances="always"
    )


class Noul(_DecisionData):
    """Ask for the probability that a statement is true.

    The answer is a probability, not a boolean action. Application code chooses
    any threshold. OpenAI calls this operation `Predicate` on its native interface.
    """

    type: Literal["noul"] = "noul"
    instructions: str = Field(min_length=1)
    """Complete proposition to evaluate against the state."""


class DecisionOption(_DecisionData):
    """One categorical alternative, preserving string versus boolean identity."""

    value: str | bool
    description: str | None = None


class Choice(_DecisionData):
    """Select among defined, unordered alternatives.

    Boolean alternatives require provider support. Options are records rather than
    JSON object keys so `True` and `"true"` remain distinct through serialization.
    """

    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1)
    options: list[DecisionOption] = Field(min_length=2)

    @model_validator(mode="after")
    def _unique_options(self) -> Self:
        identities = {(type(option.value), option.value) for option in self.options}
        if len(identities) != len(self.options):
            msg = "Choice options must have distinct values."
            raise ValueError(msg)
        return self


class DecisionLevel(_DecisionData):
    """One rubric level, indexed by its position in the supplied list."""

    label: str = Field(min_length=1)
    description: str | None = None


class Score(_DecisionData):
    """Evaluate defined, ordered rubric levels.

    The answer is the expected zero-based level index and may be fractional.
    """

    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1)
    levels: list[DecisionLevel] = Field(min_length=2)


Question: TypeAlias = Annotated[Noul | Choice | Score, Field(discriminator="type")]
"""Canonical question union; dictionaries use `noul`, `choice`, or `score`."""


class DecisionRequest(TypedDict):
    """Complete input for one judgment over shared state.

    Questions are independent. A dependent question belongs in a later invocation.
    Equivalent question dictionaries are validated at the invocation boundary.
    """

    state: State
    questions: Annotated[dict[str, Question], Field(min_length=1)]


class _AnswerData(_DecisionData):
    abstained: bool | None = None
    """Provider-reported abstention; `None` means unreported."""
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    """Provider-specific diagnostics and confidence provenance."""


class NoulAnswer(_AnswerData):
    """Probability that a `Noul` proposition is true, without thresholding."""

    type: Literal["noul"] = "noul"
    probability: _Probability


class ChoiceProbability(_DecisionData):
    """Probability assigned to one typed categorical value."""

    value: str | bool
    probability: _Probability


class ChoiceAnswer(_AnswerData):
    """Selected alternative and the complete categorical distribution."""

    type: Literal["choice"] = "choice"
    value: str | bool
    probabilities: list[ChoiceProbability] = Field(min_length=2)
    provider_confidence: _Probability | None = None
    """Native confidence; its formula and thresholds are provider-specific."""

    @property
    def selected_probability(self) -> float:
        """Return the selected value's probability, not a guarantee of accuracy."""
        for item in self.probabilities:
            if type(item.value) is type(self.value) and item.value == self.value:
                return item.probability
        msg = "Selected choice is absent from its probability distribution."
        raise ValueError(msg)


class ScoreAnswer(_AnswerData):
    """Expected zero-based rubric index and probabilities in request order."""

    type: Literal["score"] = "score"
    score: Annotated[float, Field(strict=True, allow_inf_nan=False)]
    levels: list[DecisionLevel] = Field(min_length=2)
    probabilities: list[_Probability] = Field(min_length=2)
    provider_confidence: _Probability | None = None


class RefusalAnswer(_AnswerData):
    """Explicit provider refusal, distinct from failure or a fallback option."""

    type: Literal["refusal"] = "refusal"


Answer: TypeAlias = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer,
    Field(discriminator="type"),
]
"""Canonical answer union, narrowed by its `type` discriminator."""


class DecisionUsage(_DecisionData):
    """Reported token counts; absent counts remain unknown."""

    input_tokens: _Count | None = None
    output_tokens: _Count | None = None

    @property
    def total_tokens(self) -> int | None:
        """Return the total only when both components were reported."""
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


class DecisionResponse(_DecisionData):
    """Typed answers keyed by question ID, plus reported identity and usage."""

    answers: dict[str, Answer]
    model: str | None = None
    usage: DecisionUsage = Field(default_factory=DecisionUsage)
    response_metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionModelProfile(TypedDict, total=False):
    """Known capabilities; missing fields are unknown, not guaranteed support."""

    question_types: list[Literal["noul", "choice", "score"]]
    state_types: list[Literal["text", "json", "messages"]]
    boolean_choices: bool
    max_questions: int
    max_options: int
