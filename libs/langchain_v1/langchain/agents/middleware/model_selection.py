"""Record the model selected for an agent.

After every model call, the agent stores a
[`ModelSpec`][langchain.agents.middleware.types.ModelSpec] for the model the call
went to under the `model_selection` state key (see
[`ModelSelection`][langchain.agents.middleware.types.ModelSelection]). The record
is informational: it lets middleware and clients see which model is in use and
detect switches, but it never chooses the model for a call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import ModelSelection, ModelSpec
from langchain.chat_models.base import _BUILTIN_PROVIDERS

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

__all__ = [
    "MODEL_SELECTION_KEY",
    "ModelSelection",
    "ModelSpec",
    "current_model_spec",
    "model_spec_from_model",
]

MODEL_SELECTION_KEY = "model_selection"
"""Agent state key holding the [`ModelSelection`][langchain.agents.middleware.types.ModelSelection]."""  # noqa: E501

SAFE_INIT_KWARGS = ("temperature", "top_p", "top_k", "max_tokens", "reasoning_effort")
"""Model attributes copied into `ModelSpec.init_kwargs`.

An allow list rather than a deny list, so credentials, headers, clients and
endpoints (which may embed tokens) are never stored in agent state.
"""

_SCALAR_TYPES = (str, int, float, bool)


def model_spec_from_model(model: BaseChatModel) -> ModelSpec | None:
    """Describe `model` as a serializable `ModelSpec`.

    Args:
        model: The chat model to describe.

    Returns:
        The spec, or `None` if the model does not report a model name.
    """
    try:
        params: Mapping[str, Any] = model._get_ls_params()  # noqa: SLF001
    except Exception:
        return None
    name = params.get("ls_model_name")
    provider = _init_provider(model, params.get("ls_provider"))
    if not isinstance(name, str) or not name or provider is None:
        return None
    spec: ModelSpec = {"provider": provider, "model": name}
    profile = getattr(model, "profile", None)
    max_input_tokens = profile.get("max_input_tokens") if isinstance(profile, Mapping) else None
    if isinstance(max_input_tokens, int) and not isinstance(max_input_tokens, bool):
        spec["max_input_tokens"] = max_input_tokens
    init_kwargs = _safe_init_kwargs({key: getattr(model, key, None) for key in SAFE_INIT_KWARGS})
    if init_kwargs:
        spec["init_kwargs"] = init_kwargs
    return spec


def current_model_spec(state: Mapping[str, Any]) -> ModelSpec | None:
    """Return the currently selected model recorded in agent `state`, if any.

    Args:
        state: Agent state.

    Returns:
        `state["model_selection"]["current"]`, or `None` when nothing is recorded.
    """
    selection: ModelSelection | None = state.get(MODEL_SELECTION_KEY)
    return selection.get("current") if selection else None


def _safe_init_kwargs(values: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        key: value
        for key, value in values.items()
        if key in SAFE_INIT_KWARGS and isinstance(value, _SCALAR_TYPES)
    }


def _init_provider(model: BaseChatModel, ls_provider: object) -> str | None:
    """Map a model's class back to its `init_chat_model` provider key."""
    cls = type(model)
    package = cls.__module__.split(".", maxsplit=1)[0]
    candidates = [
        provider
        for provider, (module, class_name, _) in _BUILTIN_PROVIDERS.items()
        if class_name == cls.__name__ and module.split(".", maxsplit=1)[0] == package
    ]
    if (
        isinstance(ls_provider, str)
        and ls_provider
        and (ls_provider in candidates or not candidates)
    ):
        return ls_provider
    return candidates[0] if candidates else None
