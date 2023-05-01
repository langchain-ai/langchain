"""Integration tests for the langchain tracer module."""
import asyncio

import pytest

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI


@pytest.mark.asyncio
async def test_openai_callback() -> None:
    llm = OpenAI(temperature=0)
    with get_openai_callback() as cb:
        llm("What is the square root of 4?")

    total_tokens = cb.total_tokens
    assert total_tokens > 0

    with get_openai_callback() as cb:
        llm("What is the square root of 4?")
        llm("What is the square root of 4?")

    assert cb.total_tokens == total_tokens * 2

    with get_openai_callback() as cb:
        await asyncio.gather(
            *[llm.agenerate(["What is the square root of 4?"]) for _ in range(3)]
        )

    assert cb.total_tokens == total_tokens * 3

    task = asyncio.create_task(llm.agenerate(["What is the square root of 4?"]))
    with get_openai_callback() as cb:
        await llm.agenerate(["What is the square root of 4?"])

    await task
    assert cb.total_tokens == total_tokens


def test_openai_callback_agent() -> None:
    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    with get_openai_callback() as cb:
        agent.run(
            "Who is Olivia Wilde's boyfriend? "
            "What is his current age raised to the 0.23 power?"
        )
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
