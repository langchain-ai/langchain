import asyncio
import os
import time

import pytest
from aiohttp import ClientSession

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.llms import OpenAI

questions = [
    "Who won the US Open men's final in 2019? What is his age raised to the 0.334 power?",
    "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?",
    "Who won the most recent formula 1 grand prix? What is their age raised to the 0.23 power?",
    "Who won the US Open women's final in 2019? What is her age raised to the 0.34 power?",
    "Who is Beyonce's husband? What is his age raised to the 0.19 power?",
]


def test_tracing_sequential():
    os.environ["LANGCHAIN_TRACING"] = "true"

    def generate_serially():
        for q in questions[:3]:
            llm = OpenAI(temperature=0)
            tools = load_tools(["llm-math", "serpapi"], llm=llm)
            agent = initialize_agent(
                tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
            )
            agent.run(q)

    s = time.perf_counter()
    generate_serially()
    elapsed = time.perf_counter() - s
    print(f"Serial executed in {elapsed:0.2f} seconds.")


@pytest.mark.asyncio
async def test_tracing_concurrent():
    os.environ["LANGCHAIN_TRACING"] = "true"
    aiosession = ClientSession()
    llm = OpenAI(temperature=0)
    async_tools = load_tools(["llm-math", "serpapi"], llm=llm, aiosession=aiosession)
    agent = initialize_agent(
        async_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    tasks = [agent.arun(q) for q in questions[:3]]
    await asyncio.gather(*tasks)
    await aiosession.close()
