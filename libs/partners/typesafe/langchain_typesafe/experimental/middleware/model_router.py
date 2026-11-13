"""Experimental model-routing middleware powered by TypeSafe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

try:
    from langchain.agents.middleware.model_router import (
        ModelChoice,
    )
    from langchain.agents.middleware.model_router import (
        ModelRouterMiddleware as _ModelRouterMiddleware,
    )
except ImportError as error:
    msg = (
        "ModelRouterMiddleware requires the LangChain agent framework. "
        "Install it with `pip install 'langchain-typesafe[experimental]'`."
    )
    raise ImportError(msg) from error

from langchain_core._api import deprecated

from langchain_typesafe.classifier import TypeSafeClassifier

if TYPE_CHECKING:
    from langchain_typesafe.types import Content


@deprecated(
    since="0.1.0",
    removal="0.2.0",
    message=(
        "`langchain_typesafe.experimental.middleware.ModelRouterMiddleware` is "
        "deprecated since langchain-typesafe 0.1.0 and will be removed in 0.2.0. Use "
        "`langchain.agents.middleware.ModelRouterMiddleware` with "
        "`classifier=TypeSafeClassifier()` instead."
    ),
    package="langchain-typesafe",
)
class ModelRouterMiddleware(_ModelRouterMiddleware):
    """Select an agent's model with a TypeSafe `Choice` classification.

    This is `langchain.agents.middleware.ModelRouterMiddleware` with a default
    `TypeSafeClassifier`. The middleware classifies the latest human message once
    before an agent run, stores the complete `ChoiceAnswer` in agent state, and uses its
    selected label for every model call in the run. Classifier failures propagate and
    terminate the run rather than silently selecting a different model.

    Install the experimental extra to use this class:

    ```bash
    pip install "langchain-typesafe[experimental]"
    ```

    Args:
        choices: Named model choices, each containing a LangChain model or model
            string and the criterion for selecting it.
        instructions: Additional instructions TypeSafe should follow when selecting a
            route.
        classifier: Optional `TypeSafeClassifier` used for routing, for example one
            configured with a custom `base_url`, `client`, or `async_client`. If
            omitted, a default `TypeSafeClassifier` is created.

    Raises:
        pydantic.ValidationError: If no model choices are provided.

    ??? example "Route agent calls by task"

        ```python
        from langchain.agents import create_agent
        from langchain_typesafe.experimental.middleware import (
            ModelChoice,
            ModelRouterMiddleware,
        )

        router = ModelRouterMiddleware(
            choices={
                "fast": ModelChoice(
                    model="openai:gpt-5-mini",
                    criteria="Simple, well-scoped tasks.",
                ),
                "powerful": ModelChoice(
                    model=powerful_model,
                    criteria="Complex tasks requiring deeper reasoning.",
                ),
            },
            instructions="Choose the least costly model suited to the task.",
        )
        agent = create_agent("openai:gpt-5-mini", middleware=[router])
        ```
    """

    classifier: TypeSafeClassifier

    def __init__(
        self,
        *,
        choices: Mapping[str, ModelChoice],
        instructions: Content,
        classifier: TypeSafeClassifier | None = None,
    ) -> None:
        """Initialize the model router."""
        super().__init__(
            choices=choices,
            instructions=instructions,
            classifier=classifier if classifier is not None else TypeSafeClassifier(),
        )


__all__ = ["ModelChoice", "ModelRouterMiddleware"]
