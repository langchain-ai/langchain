"""Load agent."""

import contextlib
from collections.abc import Sequence
from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from langchain_classic._api.deprecation import AGENT_DEPRECATION_WARNING
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents.loading import load_agent
from langchain_classic.agents.types import AGENT_TO_CLASS


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
def initialize_agent(
    tools: Sequence[BaseTool],
    llm: BaseLanguageModel,
    agent: AgentType | None = None,
    callback_manager: BaseCallbackManager | None = None,
    agent_path: str | None = None,
    agent_kwargs: dict | None = None,
    *,
    tags: Sequence[str] | None = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Load an agent executor given tools and LLM.

    Args:
        tools: List of tools this agent has access to.
        llm: Language model to use as the agent.
        agent: Agent type to use. If `None` and agent_path is also None, will default
            to AgentType.ZERO_SHOT_REACT_DESCRIPTION.
        callback_manager: CallbackManager to use. Global callback manager is used if
            not provided.
        agent_path: Path to serialized agent to use. If `None` and agent is also None,
            will default to AgentType.ZERO_SHOT_REACT_DESCRIPTION.
        agent_kwargs: Additional keyword arguments to pass to the underlying agent.

        tags: Tags to apply to the traced runs.
        kwargs: Additional keyword arguments passed to the agent executor.

    Returns:
        An agent executor.

    Raises:
        ValueError: If both `agent` and `agent_path` are specified.
        ValueError: If `agent` is not a valid agent type.
        ValueError: If both `agent` and `agent_path` are None.
    """
    tags_ = list(tags) if tags else []
    if agent is None and agent_path is None:
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION
    if agent is not None and agent_path is not None:
        msg = (
            "Both `agent` and `agent_path` are specified, "
            "but at most only one should be."
        )
        raise ValueError(msg)
    if agent is not None:
        if agent not in AGENT_TO_CLASS:
            msg = (
                f"Got unknown agent type: {agent}. "
                f"Valid types are: {AGENT_TO_CLASS.keys()}."
            )
            raise ValueError(msg)
        tags_.append(agent.value if isinstance(agent, AgentType) else agent)
        agent_cls = AGENT_TO_CLASS[agent]
        agent_kwargs = agent_kwargs or {}
        agent_obj = agent_cls.from_llm_and_tools(
            llm,
            tools,
            callback_manager=callback_manager,
            **agent_kwargs,
        )
    elif agent_path is not None:
        agent_obj = load_agent(
            agent_path,
            llm=llm,
            tools=tools,
            callback_manager=callback_manager,
        )
        with contextlib.suppress(NotImplementedError):
            # TODO: Add tags from the serialized object directly.
            tags_.append(agent_obj._agent_type)  # noqa: SLF001
    else:
        msg = (
            "Somehow both `agent` and `agent_path` are None, this should never happen."
        )
        raise ValueError(msg)
    return AgentExecutor.from_agent_and_tools(
        agent=agent_obj,
        tools=tools,
        callback_manager=callback_manager,
        tags=tags_,
        **kwargs,
    )
