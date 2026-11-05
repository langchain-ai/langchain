"""Record and rebuild the model selected for an agent.

The agent stores a [`ModelSpec`][langchain.agents.middleware.types.ModelSpec] for
the model each call goes to under the `model_selection` state key (see
[`ModelSelection`][langchain.agents.middleware.types.ModelSelection]), so
middleware such as `SummarizationMiddleware` can follow the current model from
any hook, regardless of middleware order.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import ModelSelection, ModelSpec
from langchain.chat_models.base import _BUILTIN_PROVIDERS, init_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

__all__ = [
    "MODEL_SELECTION_KEY",
    "ModelSelection",
    "ModelSpec",
    "current_model_spec",
    "init_model_from_spec",
    "model_spec_from_model",
]

MODEL_SELECTION_KEY = "model_selection"
"""Agent state key holding the [`ModelSelection`][langchain.agents.middleware.types.ModelSelection]."""  # noqa: E501

SAFE_INIT_KWARGS = ("temperature", "top_p", "top_k", "max_tokens", "reasoning_effort")
"""Model attributes copied into `ModelSpec.init_kwargs`.

An allow list rather than a deny list, so credentials, headers, clients and
endpoints (which may embed tokens, or redirect a rebuilt client's API key) are
never stored or used to rebuild a model.
"""

_SCALAR_TYPES = (str, int, float, bool)
_REBUILT_MODEL_CACHE_SIZE = 32


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


def init_model_from_spec(spec: ModelSpec) -> BaseChatModel:
    """Build (or reuse) a chat model for `spec` with `init_chat_model`.

    Only `init_chat_model`'s built-in providers and `SAFE_INIT_KWARGS` are used, so a
    spec written to state cannot point a rebuilt client, and its credentials, at
    another endpoint. Credentials come from the environment, as with
    `init_chat_model`.

    Rebuilt models are cached in a bounded, thread-safe LRU keyed by the spec's
    canonical JSON (never by object `id()`), so repeated calls reuse one client.

    Args:
        spec: The model spec.

    Returns:
        The chat model.

    Raises:
        ValueError: If the spec's provider is not a built-in `init_chat_model`
            provider.
    """
    provider = spec.get("provider")
    if provider not in _BUILTIN_PROVIDERS:
        msg = (
            f"Cannot rebuild a chat model for provider {provider!r}: only "
            "`init_chat_model`'s built-in providers can be rebuilt from agent state."
        )
        raise ValueError(msg)
    key = {
        "provider": provider,
        "model": spec["model"],
        "init_kwargs": _safe_init_kwargs(spec.get("init_kwargs") or {}),
    }
    return _init_cached(json.dumps(key, sort_keys=True))


@lru_cache(maxsize=_REBUILT_MODEL_CACHE_SIZE)
def _init_cached(key: str) -> BaseChatModel:
    spec = json.loads(key)
    model: BaseChatModel = init_chat_model(
        str(spec["model"]), model_provider=str(spec["provider"]), **spec["init_kwargs"]
    )
    return model


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
