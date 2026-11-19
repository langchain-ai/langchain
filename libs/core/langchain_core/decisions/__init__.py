"""Decision models: typed probabilistic answers about text, messages, or state."""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.decisions.base import BaseDecisionModel
    from langchain_core.decisions.types import (
        Answer,
        Box,
        BoxAnswer,
        BoxCoordinates,
        Choice,
        ChoiceAnswer,
        ClassifierRequest,
        ClassifierResponse,
        Content,
        Noul,
        NoulAnswer,
        NoulCriteria,
        Question,
        Score,
        ScoreAnswer,
        State,
    )

__all__ = (
    "Answer",
    "BaseDecisionModel",
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
)

_dynamic_imports = {
    "Answer": "types",
    "BaseDecisionModel": "base",
    "Box": "types",
    "BoxAnswer": "types",
    "BoxCoordinates": "types",
    "Choice": "types",
    "ChoiceAnswer": "types",
    "ClassifierRequest": "types",
    "ClassifierResponse": "types",
    "Content": "types",
    "Noul": "types",
    "NoulAnswer": "types",
    "NoulCriteria": "types",
    "Question": "types",
    "Score": "types",
    "ScoreAnswer": "types",
    "State": "types",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
