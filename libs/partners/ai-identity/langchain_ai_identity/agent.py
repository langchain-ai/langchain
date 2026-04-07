"""Factory for creating an AI Identity-governed LangChain agent.

Assembles a fully wired agent with AI Identity gateway enforcement and
audit logging using langgraph's ``create_react_agent``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from langchain_ai_identity._gateway import _DEFAULT_TIMEOUT
from langchain_ai_identity.callback import AIIdentityCallbackHandler
from langchain_ai_identity.chat_models import AIIdentityChatOpenAI
from langchain_ai_identity.tools import AIIdentityToolkit


def create_ai_identity_agent(
    api_key: str,
    agent_id: str,
    tools: list[BaseTool],
    *,
    model: str = "gpt-4o",
    openai_api_key: str | None = None,
    system_message: str = "You are a helpful assistant.",
    fail_closed: bool = True,
    timeout: float = _DEFAULT_TIMEOUT,
    gateway_url: str | None = None,
    api_url: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create an AI Identity-governed agent with full policy enforcement.

    Constructs an :class:`AIIdentityChatOpenAI` model, wraps tools via
    :class:`AIIdentityToolkit`, attaches an
    :class:`AIIdentityCallbackHandler`, and returns a ready-to-use
    agent built with ``langgraph.prebuilt.create_react_agent``.

    Args:
        api_key: AI Identity API key (``aid_sk_...``).
        agent_id: Unique identifier for the agent.
        tools: LangChain tools the agent may invoke.
        model: OpenAI model name. Defaults to ``"gpt-4o"``.
        openai_api_key: OpenAI API key.  Falls back to the
            ``OPENAI_API_KEY`` environment variable.
        system_message: System prompt for the agent.
        fail_closed: Block on deny when ``True`` (default).
        timeout: HTTP timeout for gateway / audit calls in seconds.
        gateway_url: Override for the AI Identity gateway base URL.
        api_url: Override for the AI Identity API base URL.
        **kwargs: Additional keyword arguments forwarded to
            ``create_react_agent``.

    Returns:
        A configured langgraph agent.
    """
    llm_kwargs: dict[str, Any] = {
        "model": model,
        "agent_id": agent_id,
        "ai_identity_api_key": api_key,
        "fail_closed": fail_closed,
        "ai_identity_timeout": timeout,
        "gateway_url": gateway_url,
    }
    if openai_api_key is not None:
        llm_kwargs["openai_api_key"] = openai_api_key

    llm = AIIdentityChatOpenAI(**llm_kwargs)

    toolkit = AIIdentityToolkit(
        api_key=api_key,
        agent_id=agent_id,
        tools=tools,
        fail_closed=fail_closed,
        timeout=timeout,
        gateway_url=gateway_url,
    )
    governed_tools = toolkit.get_tools()

    AIIdentityCallbackHandler(
        api_key=api_key,
        agent_id=agent_id,
        fail_closed=fail_closed,
        timeout=timeout,
        api_url=api_url,
    )

    return create_react_agent(
        model=llm,
        tools=governed_tools,
        prompt=system_message,
        **kwargs,
    )
