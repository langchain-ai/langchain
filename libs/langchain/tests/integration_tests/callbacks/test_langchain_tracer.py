"""Integration tests for the langchain tracer module."""
import asyncio
import os

import pytest
from aiohttp import ClientSession

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import tracing_enabled
from langchain.callbacks.manager import (
    atrace_as_chain_group,
    trace_as_chain_group,
    tracing_v2_enabled,
)
from langchain.chains import LLMChain
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

questions = [
    (
        "Who won the US Open men's final in 2019? "
        "What is his age raised to the 0.334 power?"
    ),
    (
        "Who is Olivia Wilde's boyfriend? "
        "What is his current age raised to the 0.23 power?"
    ),
    (
        "Who won the most recent formula 1 grand prix? "
        "What is their age raised to the 0.23 power?"
    ),
    (
        "Who won the US Open women's final in 2019? "
        "What is her age raised to the 0.34 power?"
    ),
    ("Who is Beyonce's husband? " "What is his age raised to the 0.19 power?"),
]


def test_tracing_sequential() -> None:
    os.environ["LANGCHAIN_TRACING"] = "true"

    for q in questions[:3]:
        llm = OpenAI(temperature=0)
        tools = load_tools(["llm-math", "serpapi"], llm=llm)
        agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )
        agent.run(q)


def test_tracing_session_env_var() -> None:
    os.environ["LANGCHAIN_TRACING"] = "true"
    os.environ["LANGCHAIN_SESSION"] = "my_session"

    llm = OpenAI(temperature=0)
    tools = load_tools(["llm-math", "serpapi"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    agent.run(questions[0])
    if "LANGCHAIN_SESSION" in os.environ:
        del os.environ["LANGCHAIN_SESSION"]


@pytest.mark.asyncio
async def test_tracing_concurrent() -> None:
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


@pytest.mark.asyncio
async def test_tracing_concurrent_bw_compat_environ() -> None:
    os.environ["LANGCHAIN_HANDLER"] = "langchain"
    if "LANGCHAIN_TRACING" in os.environ:
        del os.environ["LANGCHAIN_TRACING"]
    aiosession = ClientSession()
    llm = OpenAI(temperature=0)
    async_tools = load_tools(["llm-math", "serpapi"], llm=llm, aiosession=aiosession)
    agent = initialize_agent(
        async_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    tasks = [agent.arun(q) for q in questions[:3]]
    await asyncio.gather(*tasks)
    await aiosession.close()
    if "LANGCHAIN_HANDLER" in os.environ:
        del os.environ["LANGCHAIN_HANDLER"]


def test_tracing_context_manager() -> None:
    llm = OpenAI(temperature=0)
    tools = load_tools(["llm-math", "serpapi"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    if "LANGCHAIN_TRACING" in os.environ:
        del os.environ["LANGCHAIN_TRACING"]
    with tracing_enabled() as session:
        assert session
        agent.run(questions[0])  # this should be traced

    agent.run(questions[0])  # this should not be traced


@pytest.mark.asyncio
async def test_tracing_context_manager_async() -> None:
    llm = OpenAI(temperature=0)
    async_tools = load_tools(["llm-math", "serpapi"], llm=llm)
    agent = initialize_agent(
        async_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    if "LANGCHAIN_TRACING" in os.environ:
        del os.environ["LANGCHAIN_TRACING"]

    # start a background task
    task = asyncio.create_task(agent.arun(questions[0]))  # this should not be traced
    with tracing_enabled() as session:
        assert session
        tasks = [agent.arun(q) for q in questions[1:4]]  # these should be traced
        await asyncio.gather(*tasks)

    await task


@pytest.mark.asyncio
async def test_tracing_v2_environment_variable() -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    aiosession = ClientSession()
    llm = OpenAI(temperature=0)
    async_tools = load_tools(["llm-math", "serpapi"], llm=llm, aiosession=aiosession)
    agent = initialize_agent(
        async_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    tasks = [agent.arun(q) for q in questions[:3]]
    await asyncio.gather(*tasks)
    await aiosession.close()


def test_tracing_v2_context_manager() -> None:
    llm = ChatOpenAI(temperature=0)
    tools = load_tools(["llm-math", "serpapi"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    if "LANGCHAIN_TRACING_V2" in os.environ:
        del os.environ["LANGCHAIN_TRACING_V2"]
    with tracing_v2_enabled():
        agent.run(questions[0])  # this should be traced

    agent.run(questions[0])  # this should not be traced


def test_tracing_v2_chain_with_tags() -> None:
    llm = OpenAI(temperature=0)
    chain = ConstitutionalChain.from_llm(
        llm,
        chain=LLMChain.from_string(llm, "Q: {question} A:"),
        tags=["only-root"],
        constitutional_principles=[
            ConstitutionalPrinciple(
                critique_request="Tell if this answer is good.",
                revision_request="Give a better answer.",
            )
        ],
    )
    if "LANGCHAIN_TRACING_V2" in os.environ:
        del os.environ["LANGCHAIN_TRACING_V2"]
    with tracing_v2_enabled():
        chain.run("what is the meaning of life", tags=["a-tag"])


def test_tracing_v2_agent_with_metadata() -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    llm = OpenAI(temperature=0)
    chat = ChatOpenAI(temperature=0)
    tools = load_tools(["llm-math", "serpapi"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    chat_agent = initialize_agent(
        tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    agent.run(questions[0], tags=["a-tag"], metadata={"a": "b", "c": "d"})
    chat_agent.run(questions[0], tags=["a-tag"], metadata={"a": "b", "c": "d"})


@pytest.mark.asyncio
async def test_tracing_v2_async_agent_with_metadata() -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    llm = OpenAI(temperature=0, metadata={"f": "g", "h": "i"})
    chat = ChatOpenAI(temperature=0, metadata={"f": "g", "h": "i"})
    async_tools = load_tools(["llm-math", "serpapi"], llm=llm)
    agent = initialize_agent(
        async_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    chat_agent = initialize_agent(
        async_tools,
        chat,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    await agent.arun(questions[0], tags=["a-tag"], metadata={"a": "b", "c": "d"})
    await chat_agent.arun(questions[0], tags=["a-tag"], metadata={"a": "b", "c": "d"})


def test_trace_as_group() -> None:
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    with trace_as_chain_group("my_group", inputs={"input": "cars"}) as group_manager:
        chain.run(product="cars", callbacks=group_manager)
        chain.run(product="computers", callbacks=group_manager)
        final_res = chain.run(product="toys", callbacks=group_manager)
        group_manager.on_chain_end({"output": final_res})

    with trace_as_chain_group("my_group_2", inputs={"input": "toys"}) as group_manager:
        final_res = chain.run(product="toys", callbacks=group_manager)
        group_manager.on_chain_end({"output": final_res})


def test_trace_as_group_with_env_set() -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    with trace_as_chain_group(
        "my_group_env_set", inputs={"input": "cars"}
    ) as group_manager:
        chain.run(product="cars", callbacks=group_manager)
        chain.run(product="computers", callbacks=group_manager)
        final_res = chain.run(product="toys", callbacks=group_manager)
        group_manager.on_chain_end({"output": final_res})

    with trace_as_chain_group(
        "my_group_2_env_set", inputs={"input": "toys"}
    ) as group_manager:
        final_res = chain.run(product="toys", callbacks=group_manager)
        group_manager.on_chain_end({"output": final_res})


@pytest.mark.asyncio
async def test_trace_as_group_async() -> None:
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    async with atrace_as_chain_group("my_async_group") as group_manager:
        await chain.arun(product="cars", callbacks=group_manager)
        await chain.arun(product="computers", callbacks=group_manager)
        await chain.arun(product="toys", callbacks=group_manager)

    async with atrace_as_chain_group(
        "my_async_group_2", inputs={"input": "toys"}
    ) as group_manager:
        res = await asyncio.gather(
            *[
                chain.arun(product="toys", callbacks=group_manager),
                chain.arun(product="computers", callbacks=group_manager),
                chain.arun(product="cars", callbacks=group_manager),
            ]
        )
        await group_manager.on_chain_end({"output": res})
