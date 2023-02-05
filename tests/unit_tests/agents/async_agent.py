import asyncio
from langchain.agents import initialize_agent, load_tools
from langchain.llms import OpenAI
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.tracers import LangChainTracer

QUESTION = "Who won the US Open men's final in 2019? What is his age raised to the 0.334 power?"


def generate_serially():
    for _ in range(4):
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager = CallbackManager([StdOutCallbackHandler(), tracer])
        llm = OpenAI(temperature=0, callback_manager=manager)
        tools = load_tools(["llm-math", "serpapi"], llm=llm)
        agent = initialize_agent(
            tools, llm, agent="zero-shot-react-description", verbose=True, callback_manager=manager
        )
        agent.run(QUESTION)


async def generate_concurrently():
    agents = []
    for color in ["blue", "green", "red", "pink"]:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager = CallbackManager([StdOutCallbackHandler(color=color), tracer])
        llm = OpenAI(temperature=0, callback_manager=manager)
        async_tools = load_tools(["llm-math", "serpapi"], llm=llm)
        agents.append(
            initialize_agent(async_tools, llm, agent="zero-shot-react-description", verbose=True, callback_manager=manager)
        )
    tasks = [async_agent.arun(QUESTION) for async_agent in agents]
    await asyncio.gather(*tasks)


import time

# s = time.perf_counter()
# generate_serially()
# elapsed = time.perf_counter() - s
# print(f"Serial executed in {elapsed:0.2f} seconds.")

if __name__ == "__main__":
    s = time.perf_counter()
    asyncio.run(generate_concurrently())
    elapsed = time.perf_counter() - s
    print(f"Concurrent executed in {elapsed:0.2f} seconds.")