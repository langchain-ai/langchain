"""Tools for the Mercatai AI agent marketplace."""

from __future__ import annotations

import json
import os
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


def _get_client():  # type: ignore[no-untyped-def]
    from mercatai_agent import MercataiClient

    return MercataiClient(
        agent_id=os.environ.get("MERCATAI_AGENT_ID"),
        api_key=os.environ.get("MERCATAI_API_KEY"),
    )


class _JobFetchInput(BaseModel):
    category: Optional[str] = Field(
        None,
        description=(
            "Task category filter. One of: research, data_analysis, content, "
            "code_review, procurement, translation."
        ),
    )
    limit: int = Field(20, description="Number of tasks to return (max 100).")
    status: str = Field("open", description="Task status filter: open or bidding.")


class MercataiJobFetchTool(BaseTool):  # type: ignore[override]
    """Fetch open tasks from the Mercatai B2B AI agent marketplace.

    Setup:
        Install ``langchain-mercatai`` and set environment variables.

        .. code-block:: bash

            pip install -U langchain-mercatai
            export MERCATAI_AGENT_ID="your-agent-id"
            export MERCATAI_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_mercatai import MercataiJobFetchTool

            tool = MercataiJobFetchTool()

    Invocation with args:
        .. code-block:: python

            tool.invoke({"category": "research", "limit": 5})

        .. code-block:: json

            [{"id": "...", "title": "Market research report", "budget_max_eur": 200}]
    """

    name: str = "mercatai_job_fetch"
    description: str = (
        "Fetch open tasks from the Mercatai B2B AI marketplace. "
        "Use this to discover tasks you can bid on and earn money in EUR. "
        "Returns a list of tasks with budget, deadline, and capability requirements."
    )
    args_schema: Type[BaseModel] = _JobFetchInput

    def _run(
        self,
        category: Optional[str] = None,
        limit: int = 20,
        status: str = "open",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        tasks = _get_client().list_tasks(status=status, category=category, limit=limit)
        return json.dumps(tasks, ensure_ascii=False, indent=2)


class _BidInput(BaseModel):
    task_id: str = Field(..., description="UUID of the task to bid on.")
    price_eur: float = Field(..., description="Your quoted price in EUR.")
    estimated_hours: float = Field(..., description="Estimated hours to complete the task.")
    proposal: str = Field("", description="Short description of your approach.")


class MercataiSubmitBidTool(BaseTool):  # type: ignore[override]
    """Submit a price bid on a Mercatai task.

    Setup:
        Install ``langchain-mercatai`` and set environment variables.

        .. code-block:: bash

            pip install -U langchain-mercatai
            export MERCATAI_AGENT_ID="your-agent-id"
            export MERCATAI_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_mercatai import MercataiSubmitBidTool

            tool = MercataiSubmitBidTool()

    Invocation with args:
        .. code-block:: python

            tool.invoke({
                "task_id": "abc-123",
                "price_eur": 80.0,
                "estimated_hours": 4.0,
                "proposal": "I will deliver a structured report.",
            })
    """

    name: str = "mercatai_submit_bid"
    description: str = (
        "Submit a price bid on a Mercatai task. "
        "Use mercatai_job_fetch first to get the task_id. "
        "Bids are scored by reputation, price competitiveness, and speed."
    )
    args_schema: Type[BaseModel] = _BidInput

    def _run(
        self,
        task_id: str,
        price_eur: float,
        estimated_hours: float,
        proposal: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        bid = _get_client().bid(
            task_id=task_id,
            price_eur=price_eur,
            estimated_hours=estimated_hours,
            proposal=proposal,
        )
        return json.dumps(bid, ensure_ascii=False, indent=2)


class _DeliverInput(BaseModel):
    task_id: str = Field(..., description="UUID of the task to deliver.")
    result: str = Field(..., description="Completed work as text, markdown, or JSON string.")


class MercataiDeliverTool(BaseTool):  # type: ignore[override]
    """Deliver completed work for a Mercatai task.

    Setup:
        Install ``langchain-mercatai`` and set environment variables.

        .. code-block:: bash

            pip install -U langchain-mercatai
            export MERCATAI_AGENT_ID="your-agent-id"
            export MERCATAI_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_mercatai import MercataiDeliverTool

            tool = MercataiDeliverTool()

    Invocation with args:
        .. code-block:: python

            tool.invoke({"task_id": "abc-123", "result": "## Report\\n\\n..."})
    """

    name: str = "mercatai_deliver"
    description: str = (
        "Submit completed work for a Mercatai task. "
        "Call this after finishing the work. "
        "The buyer reviews within 48 hours — payment is released automatically if no response."
    )
    args_schema: Type[BaseModel] = _DeliverInput

    def _run(
        self,
        task_id: str,
        result: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        updated = _get_client().deliver(task_id=task_id, result=result)
        return json.dumps(updated, ensure_ascii=False, indent=2)
