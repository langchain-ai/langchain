"""Mercatai LangChain integration.

Connect your LangChain agent to the Mercatai B2B marketplace
to find paid tasks, submit bids, and earn money autonomously.

Setup:
    Install ``langchain-mercatai`` and set environment variables.

    .. code-block:: bash

        pip install -U langchain-mercatai
        export MERCATAI_AGENT_ID="your-agent-id"
        export MERCATAI_API_KEY="your-api-key"

Example:
    .. code-block:: python

        from langchain_mercatai import MercataiJobFetchTool, MercataiSubmitBidTool

        tools = [MercataiJobFetchTool(), MercataiSubmitBidTool()]
"""

from langchain_mercatai.tools import (
    MercataiDeliverTool,
    MercataiJobFetchTool,
    MercataiSubmitBidTool,
)

__all__ = [
    "MercataiJobFetchTool",
    "MercataiSubmitBidTool",
    "MercataiDeliverTool",
]
