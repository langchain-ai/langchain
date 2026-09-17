"""LangChain agent middleware powered by TypeSafe classifiers.

This package offers agent middleware that make decisions with TypeSafe's classifiers:
routing a run to a model, selecting skills for a request, and gating risky tool calls.

Requests go through the official TypeSafe Python SDK, which owns the wire protocol,
question and answer types, retries, and response validation. This package does not
wrap or re-export them — import them from `typesafe_sdk` when configuring a
middleware, and use its exception types, which every error raised here also subclasses
alongside the matching `langchain_core.exceptions.ModelError`.

!!! warning

    Every middleware here is experimental. Their APIs may change without notice.
"""

from langchain_typesafe._version import __version__
from langchain_typesafe.middleware import (
    AutoModeMiddleware,
    ModelChoice,
    ModelRouterMiddleware,
    Skill,
    SkillsMiddleware,
    SkillSource,
)

__all__ = [
    "AutoModeMiddleware",
    "ModelChoice",
    "ModelRouterMiddleware",
    "Skill",
    "SkillSource",
    "SkillsMiddleware",
    "__version__",
]
