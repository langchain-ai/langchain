"""Experimental tool-risk middleware powered by TypeSafe."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

try:
    from langchain.agents.middleware.auto_mode import (
        _DEFAULT_INSTRUCTIONS,
    )
    from langchain.agents.middleware.auto_mode import (
        AutoModeMiddleware as _AutoModeMiddleware,
    )
except ImportError as error:
    message = (
        "AutoModeMiddleware requires the LangChain agent framework. "
        "Install it with `pip install 'langchain-typesafe[experimental]'`."
    )
    raise ImportError(message) from error

from langchain_core._api import deprecated
from langchain_core.tools import BaseTool

from langchain_typesafe.classifier import TypeSafeClassifier

if TYPE_CHECKING:
    from langchain_typesafe.types import NoulCriteria


@deprecated(
    since="0.1.0",
    removal="0.2.0",
    message=(
        "`langchain_typesafe.experimental.middleware.AutoModeMiddleware` is deprecated "
        "since langchain-typesafe 0.1.0 and will be removed in 0.2.0. Use "
        "`langchain.agents.middleware.AutoModeMiddleware` with "
        "`classifier=TypeSafeClassifier()` instead."
    ),
    package="langchain-typesafe",
)
class AutoModeMiddleware(_AutoModeMiddleware):
    """Allow low-risk tool calls and block risky calls using TypeSafe.

    This is `langchain.agents.middleware.AutoModeMiddleware` with a default
    `TypeSafeClassifier`. It intercepts explicitly configured tools immediately before
    execution and asks a TypeSafe `Noul` question for the probability that each call is
    risky or insufficiently authorized. Calls below 0.5 execute normally. Calls at or
    above it return an error `ToolMessage` without invoking the tool handler. Tool
    names not listed in `tools` bypass classification.

    The classifier receives the proposed tool call and up to 30 recent messages.
    Classification failures propagate and the tool handler is not called. This
    middleware blocks calls; it does not request human approval.

    Install the experimental extra to use this class:

    ```bash
    pip install "langchain-typesafe[experimental]"
    ```

    Args:
        tools: Tool names or `BaseTool` instances to classify before execution. Unlisted
            tools are passed to the handler without classification.
        instructions: Risk-classification instructions sent to TypeSafe.
        criteria: Optional descriptions of the blocked and allowed outcomes.
        classifier: Optional `TypeSafeClassifier` used for risk classification, for
            example one configured with a custom `base_url`, `client`, or
            `async_client`. If omitted, a default `TypeSafeClassifier` is created.

    ??? example "Customize the risk criteria"

        ```python
        from langchain.agents import create_agent
        from langchain_typesafe import NoulCriteria
        from langchain_typesafe.experimental.middleware import AutoModeMiddleware

        auto_mode = AutoModeMiddleware(
            tools=[delete_file],
            criteria=NoulCriteria(
                true="The call writes, deletes, publishes, or changes access.",
                false="The call only reads public or user-provided data.",
            ),
        )
        agent = create_agent(
            model,
            tools=[read_file, delete_file],
            middleware=[auto_mode],
        )
        ```
    """

    classifier: TypeSafeClassifier

    def __init__(
        self,
        *,
        tools: Sequence[str | BaseTool],
        instructions: str = _DEFAULT_INSTRUCTIONS,
        criteria: NoulCriteria | None = None,
        classifier: TypeSafeClassifier | None = None,
    ) -> None:
        """Initialize the tool-risk middleware.

        Args:
            tools: Tool names or instances to classify before execution.
            instructions: Risk-classification instructions sent to TypeSafe.
            criteria: Descriptions of the blocked and allowed outcomes.
            classifier: `TypeSafeClassifier` to use instead of the default.

        Raises:
            pydantic.ValidationError: If the tool configuration is invalid.
        """
        super().__init__(
            tools=tools,
            instructions=instructions,
            criteria=criteria,
            classifier=classifier if classifier is not None else TypeSafeClassifier(),
        )


__all__ = ["AutoModeMiddleware"]
