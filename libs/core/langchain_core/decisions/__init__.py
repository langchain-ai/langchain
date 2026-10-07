"""Decision models: typed probabilistic answers about text, messages, or state.

!!! warning

    This module is in beta. Its API may change without notice.
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.decisions.base import BaseDecisionModel
    from langchain_core.decisions.types import (
        Answer,
        Choice,
        ChoiceAnswer,
        DecisionInput,
        DecisionRequest,
        DecisionResponse,
        Level,
        Predicate,
        PredicateAnswer,
        Question,
        RefusalAnswer,
        Score,
        ScoreAnswer,
        Usage,
    )

__all__ = (
    "Answer",
    "BaseDecisionModel",
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
)

_dynamic_imports = {
    "Answer": "types",
    "BaseDecisionModel": "base",
    "Choice": "types",
    "ChoiceAnswer": "types",
    "DecisionInput": "types",
    "DecisionRequest": "types",
    "DecisionResponse": "types",
    "Level": "types",
    "Predicate": "types",
    "PredicateAnswer": "types",
    "Question": "types",
    "RefusalAnswer": "types",
    "Score": "types",
    "ScoreAnswer": "types",
    "Usage": "types",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
