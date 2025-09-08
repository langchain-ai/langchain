"""Internal prompt resolution utilities.

This module provides utilities for resolving different types of prompt specifications
into standardized message formats for language models. It supports both synchronous
and asynchronous prompt resolution with automatic detection of callable types.

The module is designed to handle common prompt patterns across LangChain components,
particularly for summarization chains and other document processing workflows.

"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.messages import MessageLikeRepresentation
    from langgraph.runtime import Runtime

    from langchain._internal._typing import ContextT, StateT


def resolve_prompt(
    prompt: Union[
        str,
        None,
        Callable[[StateT, Runtime[ContextT]], list[MessageLikeRepresentation]],
    ],
    state: StateT,
    runtime: Runtime[ContextT],
    default_user_content: str,
    default_system_content: str,
) -> list[MessageLikeRepresentation]:
    """Resolve a prompt specification into a list of messages.

    Handles prompt resolution across different strategies. Supports callable functions,
    string system messages, and None for default behavior.

    Args:
        prompt: The prompt specification to resolve. Can be:
            - Callable: Function taking (state, runtime) returning message list.
            - str: A system message string.
            - None: Use the provided default system message.
        state: Current state, passed to callable prompts.
        runtime: LangGraph runtime instance, passed to callable prompts.
        default_user_content: User content to include (e.g., document text).
        default_system_content: Default system message when prompt is None.

    Returns:
        List of message dictionaries for language models, typically containing
        a system message and user message with content.

    Raises:
        TypeError: If prompt type is not str, None, or callable.

    Example:
        ```python
        def custom_prompt(state, runtime):
            return [{"role": "system", "content": "Custom"}]


        messages = resolve_prompt(custom_prompt, state, runtime, "content", "default")
        messages = resolve_prompt("Custom system", state, runtime, "content", "default")
        messages = resolve_prompt(None, state, runtime, "content", "Default")
        ```

    .. note::
        Callable prompts have full control over message structure and content parameter
        is ignored. String/None prompts create standard system + user structure.

    """
    if callable(prompt):
        return prompt(state, runtime)
    if isinstance(prompt, str):
        system_msg = prompt
    elif prompt is None:
        system_msg = default_system_content
    else:
        msg = f"Invalid prompt type: {type(prompt)}. Expected str, None, or callable."
        raise TypeError(msg)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": default_user_content},
    ]


async def aresolve_prompt(
    prompt: Union[
        str,
        None,
        Callable[[StateT, Runtime[ContextT]], list[MessageLikeRepresentation]],
        Callable[[StateT, Runtime[ContextT]], Awaitable[list[MessageLikeRepresentation]]],
    ],
    state: StateT,
    runtime: Runtime[ContextT],
    default_user_content: str,
    default_system_content: str,
) -> list[MessageLikeRepresentation]:
    """Async version of resolve_prompt supporting both sync and async callables.

    Handles prompt resolution across different strategies. Supports sync/async callable
    functions, string system messages, and None for default behavior.

    Args:
        prompt: The prompt specification to resolve. Can be:
            - Callable (sync): Function taking (state, runtime) returning message list.
            - Callable (async): Async function taking (state, runtime) returning
              awaitable message list.
            - str: A system message string.
            - None: Use the provided default system message.
        state: Current state, passed to callable prompts.
        runtime: LangGraph runtime instance, passed to callable prompts.
        default_user_content: User content to include (e.g., document text).
        default_system_content: Default system message when prompt is None.

    Returns:
        List of message dictionaries for language models, typically containing
        a system message and user message with content.

    Raises:
        TypeError: If prompt type is not str, None, or callable.

    Example:
        ```python
        async def async_prompt(state, runtime):
            return [{"role": "system", "content": "Async"}]


        def sync_prompt(state, runtime):
            return [{"role": "system", "content": "Sync"}]


        messages = await aresolve_prompt(async_prompt, state, runtime, "content", "default")
        messages = await aresolve_prompt(sync_prompt, state, runtime, "content", "default")
        messages = await aresolve_prompt("Custom", state, runtime, "content", "default")
        ```

    .. note::
        Callable prompts have full control over message structure and content parameter
        is ignored. Automatically detects and handles async callables.

    """
    if callable(prompt):
        result = prompt(state, runtime)
        # Check if the result is awaitable (async function)
        if inspect.isawaitable(result):
            return await result
        return result
    if isinstance(prompt, str):
        system_msg = prompt
    elif prompt is None:
        system_msg = default_system_content
    else:
        msg = f"Invalid prompt type: {type(prompt)}. Expected str, None, or callable."
        raise TypeError(msg)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": default_user_content},
    ]
