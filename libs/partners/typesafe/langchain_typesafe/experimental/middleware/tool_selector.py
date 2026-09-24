"""Experimental TypeSafe-backed tool selector middleware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import override

from langchain_typesafe.classifier import TypeSafeClassifier
from langchain_typesafe.types import Choice, ClassificationResponse, Noul

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.tools import BaseTool

_TOOL_QUESTION_PREFIX = "tool::"


@dataclass
class _SelectionRequest:
    """Prepared inputs for tool selection."""

    classifiable_tools: list[BaseTool]
    last_user_message: HumanMessage
    valid_tool_names: list[str]


class TsToolSelectorMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]
):
    """Use TypeSafe to select relevant tools before calling the main model.

    When an agent has many tools available, this middleware filters them down to only
    the most relevant ones for the user's latest request. Unlike
    `LLMToolSelectorMiddleware`, selection is not a single structured-output call to a
    text-generating model. Instead, every candidate tool becomes an independent `Noul`
    question ("is this tool needed next?"), and all questions are sent together in one
    `TypeSafeClassifier` request. A tool is kept when its answer probability is at
    least `relevance_threshold`. Since `Noul` has no native ranking, kept tools are
    ordered by that probability, descending, before `max_tools` is applied.

    !!! warning

        This middleware is experimental. Its API may change without notice.

    Install the `experimental` extra to use this class:

    ```bash
    pip install "langchain-typesafe[experimental]"
    ```

    Args:
        relevance_threshold: Minimum `Noul` probability required to keep a tool.
        max_tools: Maximum number of tools to keep. If more pass
            `relevance_threshold`, only the `max_tools` with the highest probability
            are kept. No limit if not specified.
        always_include: Tool names to always include regardless of classification.
            These do not count against `max_tools` and are not sent to TypeSafe.
        classifier_model: Model used for classification. Set to `semif-qwen3.5-4b`
            to use SemIf through a compatible gateway; defaults to Jev.

    Raises:
        ValueError: If `relevance_threshold` is not between `0` and `1`.

    ??? example "Limit to 3 tools"

        ```python
        from langchain.agents import create_agent
        from langchain_typesafe.experimental.middleware import TsToolSelectorMiddleware

        middleware = TsToolSelectorMiddleware(max_tools=3)

        agent = create_agent(
            model="openai:gpt-5.5",
            tools=[tool1, tool2, tool3, tool4, tool5],
            middleware=[middleware],
        )
        ```
    """

    def __init__(
        self,
        *,
        relevance_threshold: float = 0.5,
        max_tools: int | None = None,
        always_include: list[str] | None = None,
        classifier_model: str = "jev-latest",
    ) -> None:
        """Initialize the tool selector."""
        super().__init__()
        if not 0 <= relevance_threshold <= 1:
            msg = "`relevance_threshold` must be between 0 and 1."
            raise ValueError(msg)
        self.relevance_threshold = relevance_threshold
        self.max_tools = max_tools
        self.always_include = always_include or []
        self.classifier_model = classifier_model

    def _prepare_selection_request(
        self, request: ModelRequest[ContextT]
    ) -> _SelectionRequest | None:
        """Prepare inputs for tool selection.

        Args:
            request: The model request.

        Returns:
            `_SelectionRequest` with prepared inputs, or `None` if no selection is
                needed.

        Raises:
            ValueError: If tools in `always_include` are not found in the request.
            AssertionError: If no user message is found in the request messages.
        """
        if not request.tools:
            return None

        base_tools = [tool for tool in request.tools if not isinstance(tool, dict)]

        if self.always_include:
            available_tool_names = {tool.name for tool in base_tools}
            missing_tools = [
                name for name in self.always_include if name not in available_tool_names
            ]
            if missing_tools:
                msg = (
                    f"Tools in always_include not found in request: {missing_tools}. "
                    f"Available tools: {sorted(available_tool_names)}"
                )
                raise ValueError(msg)

        classifiable_tools = [
            tool for tool in base_tools if tool.name not in self.always_include
        ]
        if not classifiable_tools:
            return None

        last_user_message: HumanMessage
        for message in reversed(request.messages):
            if isinstance(message, HumanMessage):
                last_user_message = message
                break
        else:
            msg = "No user message found in request messages"
            raise AssertionError(msg)

        return _SelectionRequest(
            classifiable_tools=classifiable_tools,
            last_user_message=last_user_message,
            valid_tool_names=[tool.name for tool in classifiable_tools],
        )

    def _build_classifier(self, tools: list[BaseTool]) -> TypeSafeClassifier:
        """Build a classifier with one `Noul` question per candidate tool."""
        return TypeSafeClassifier(
            model=self.classifier_model,
            questions={
                f"{_TOOL_QUESTION_PREFIX}{tool.name}": Noul(
                    instructions=(
                        "Is this tool needed next to make progress on the user's "
                        f"current request? Tool: {tool.name}. "
                        f"Description: {tool.description}"
                    )
                )
                for tool in tools
            },
        )

    def _select_tool_names(
        self, response: ClassificationResponse, selection_request: _SelectionRequest
    ) -> list[str]:
        """Return classifiable tool names above threshold, ranked by probability."""
        response_nouls = response.nouls
        scored = [
            (name, response_nouls[f"{_TOOL_QUESTION_PREFIX}{name}"].noul)
            for name in selection_request.valid_tool_names
            if f"{_TOOL_QUESTION_PREFIX}{name}" in response_nouls
            and response_nouls[f"{_TOOL_QUESTION_PREFIX}{name}"].noul
            >= self.relevance_threshold
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = [name for name, _ in scored]
        if self.max_tools is not None:
            selected = selected[: self.max_tools]
        return selected

    def _classifier_config(self) -> RunnableConfig:
        """Tag classifier calls for tool selection traces."""
        return {"metadata": {"lc_source": "ts_tool_selector"}}

    def _process_selection(
        self,
        selected_tool_names: list[str],
        selection_request: _SelectionRequest,
        request: ModelRequest[ContextT],
    ) -> ModelRequest[ContextT]:
        """Return a filtered `ModelRequest` for the selected tool names."""
        tools_by_name = {
            tool.name: tool for tool in selection_request.classifiable_tools
        }
        selected_tools: list[BaseTool] = [
            tools_by_name[name] for name in selected_tool_names
        ]
        always_included_tools: list[BaseTool] = [
            tool
            for tool in request.tools
            if not isinstance(tool, dict) and tool.name in self.always_include
        ]
        provider_tools = [tool for tool in request.tools if isinstance(tool, dict)]
        return request.override(
            tools=[*selected_tools, *always_included_tools, *provider_tools]
        )

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Filter tools based on TypeSafe classification before invoking the model."""
        selection_request = self._prepare_selection_request(request)
        if selection_request is None:
            return handler(request)

        classifier = self._build_classifier(selection_request.classifiable_tools)
        response = classifier.invoke(
            selection_request.last_user_message,
            config=self._classifier_config(),
        )
        selected_tool_names = self._select_tool_names(response, selection_request)
        modified_request = self._process_selection(
            selected_tool_names, selection_request, request
        )
        return handler(modified_request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """Filter tools based on TypeSafe classification before invoking the model."""
        selection_request = self._prepare_selection_request(request)
        if selection_request is None:
            return await handler(request)

        classifier = self._build_classifier(selection_request.classifiable_tools)
        response = await classifier.ainvoke(
            selection_request.last_user_message,
            config=self._classifier_config(),
        )
        selected_tool_names = self._select_tool_names(response, selection_request)
        modified_request = self._process_selection(
            selected_tool_names, selection_request, request
        )
        return await handler(modified_request)


def _build_choice_classifier(tools: list[BaseTool], model: str) -> TypeSafeClassifier:
    """Build one categorical question for all candidate tools."""
    return TypeSafeClassifier(
        model=model,
        questions={
            "tool": Choice(
                instructions=(
                    "Which one of these tools is needed next to make progress on "
                    "the user's current request? Select exactly one tool."
                ),
                criteria={tool.name: tool.description for tool in tools},
            )
        },
    )


def _select_choice_tool_names(
    response: ClassificationResponse, selection_request: _SelectionRequest
) -> list[str]:
    """Require that the chosen tool belongs to the candidates."""
    answer = response.choices.get("tool")
    if answer is None or answer.choice not in selection_request.valid_tool_names:
        msg = "TypeSafe returned no valid tool choice for the current request"
        raise ValueError(msg)
    return [answer.choice]


class TsChoiceToolSelectorMiddleware(TsToolSelectorMiddleware):
    """Select one candidate tool for the next model call with a TypeSafe `Choice`.

    Unlike `TsToolSelectorMiddleware`, this variant asks one categorical question
    across all available candidate tools, rather than scoring each tool separately.
    The choice is made again before each model call. `always_include` tools and
    provider-specific tool definitions are passed through in addition to the chosen
    tool. Classifier failures and invalid choices raise instead of silently changing
    tool availability.

    !!! warning

        This middleware is experimental. Its API may change without notice.

    Args:
        always_include: Tool names to include without classification.
        classifier_model: Model used for classification. Set to `semif-qwen3.5-4b`
            to use SemIf through a compatible gateway; defaults to Jev.

    ??? example "Select one tool per step"

        ```python
        from langchain_typesafe.experimental.middleware import (
            TsChoiceToolSelectorMiddleware,
        )

        middleware = TsChoiceToolSelectorMiddleware()
        ```
    """

    def __init__(
        self,
        *,
        always_include: list[str] | None = None,
        classifier_model: str = "jev-latest",
    ) -> None:
        """Initialize the choice-based tool selector."""
        super().__init__(
            always_include=always_include, classifier_model=classifier_model
        )

    def _build_classifier(self, tools: list[BaseTool]) -> TypeSafeClassifier:
        """Build one categorical question for all candidate tools."""
        return _build_choice_classifier(tools, self.classifier_model)

    def _select_tool_names(
        self, response: ClassificationResponse, selection_request: _SelectionRequest
    ) -> list[str]:
        """Return only the chosen tool if it is one of the candidates."""
        return _select_choice_tool_names(response, selection_request)

    def _classifier_config(self) -> RunnableConfig:
        """Tag choice selector calls separately in traces."""
        return {"metadata": {"lc_source": "ts_choice_tool_selector"}}


class TsHybridToolSelectorMiddleware(TsToolSelectorMiddleware):
    """Choose whether to expose no tools, one tool, or several tools per model call.

    A first `Choice` question decides the selection shape from the latest human
    message. A second classifier selects tools only for `single` or `multiple`:
    `single` uses the choice selector, while `multiple` uses the thresholded
    per-tool `Noul` selector. Classifier errors and invalid choices propagate.

    !!! warning

        This middleware is experimental. Its API may change without notice.

    Args:
        relevance_threshold: Minimum probability for tools in `multiple` mode.
        max_tools: Maximum number of classified tools in `multiple` mode.
        always_include: Tool names to pass through without classification.
        classifier_model: Model used at both selection stages. Set to
            `semif-qwen3.5-4b` to use SemIf through a compatible gateway.
    """

    def __init__(
        self,
        *,
        relevance_threshold: float = 0.5,
        max_tools: int | None = None,
        always_include: list[str] | None = None,
        classifier_model: str = "jev-latest",
    ) -> None:
        """Initialize the hybrid selector with multi-tool selection settings."""
        super().__init__(
            relevance_threshold=relevance_threshold,
            max_tools=max_tools,
            always_include=always_include,
            classifier_model=classifier_model,
        )

    def _build_shape_classifier(self) -> TypeSafeClassifier:
        """Build the question that decides how many tools the next step needs."""
        return TypeSafeClassifier(
            model=self.classifier_model,
            questions={
                "shape": Choice(
                    instructions=(
                        "Does the user's current request need a tool for the next "
                        "step? Choose none if no tool is needed, single if one tool "
                        "is needed, or multiple if several tools are needed."
                    ),
                    criteria={
                        "none": "No tool is needed for the next step.",
                        "single": "Exactly one tool is needed for the next step.",
                        "multiple": "Several tools may be needed for the next step.",
                    },
                )
            },
        )

    def _shape(self, response: ClassificationResponse) -> str:
        """Require a valid answer before changing the tools available to the model."""
        answer = response.choices.get("shape")
        if answer is None or answer.choice not in {"none", "single", "multiple"}:
            msg = (
                "TypeSafe returned no valid tool selection shape "
                "for the current request"
            )
            raise ValueError(msg)
        return answer.choice

    def _classifier_config(self) -> RunnableConfig:
        """Identify second-stage tool selection separately in traces."""
        return {"metadata": {"lc_source": "ts_hybrid_tool_selector_stage_2"}}

    def _stage_one_config(self) -> RunnableConfig:
        """Identify the shape classification separately in traces."""
        return {"metadata": {"lc_source": "ts_hybrid_tool_selector_stage_1"}}

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Classify the shape and select tools before invoking the model."""
        selection_request = self._prepare_selection_request(request)
        if selection_request is None:
            return handler(request)

        shape_response = self._build_shape_classifier().invoke(
            selection_request.last_user_message, config=self._stage_one_config()
        )
        shape = self._shape(shape_response)
        selected = []
        if shape != "none":
            classifier = (
                _build_choice_classifier(
                    selection_request.classifiable_tools, self.classifier_model
                )
                if shape == "single"
                else self._build_classifier(selection_request.classifiable_tools)
            )
            response = classifier.invoke(
                selection_request.last_user_message, config=self._classifier_config()
            )
            selected = (
                _select_choice_tool_names(response, selection_request)
                if shape == "single"
                else self._select_tool_names(response, selection_request)
            )
        return handler(self._process_selection(selected, selection_request, request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Classify the shape and select tools asynchronously."""
        selection_request = self._prepare_selection_request(request)
        if selection_request is None:
            return await handler(request)

        shape_response = await self._build_shape_classifier().ainvoke(
            selection_request.last_user_message, config=self._stage_one_config()
        )
        shape = self._shape(shape_response)
        selected = []
        if shape != "none":
            classifier = (
                _build_choice_classifier(
                    selection_request.classifiable_tools, self.classifier_model
                )
                if shape == "single"
                else self._build_classifier(selection_request.classifiable_tools)
            )
            response = await classifier.ainvoke(
                selection_request.last_user_message, config=self._classifier_config()
            )
            selected = (
                _select_choice_tool_names(response, selection_request)
                if shape == "single"
                else self._select_tool_names(response, selection_request)
            )
        return await handler(
            self._process_selection(selected, selection_request, request)
        )


__all__ = [
    "TsChoiceToolSelectorMiddleware",
    "TsHybridToolSelectorMiddleware",
    "TsToolSelectorMiddleware",
]
