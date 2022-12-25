import concurrent.futures
import os
import threading

from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI


def main():
    llm = OpenAI(temperature=0)
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events",
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
    ]

    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=False
    )

    # Nested JSON tracing
    os.environ["LANGCHAIN_TRACER"] = "nested_json"

    # agent.run(
    #     "Who won the US Open men's tennis final in 2019? What is his age raised to the second power?"
    # )

    inputs = [
        "Who won the US Open men's tennis final in 2019? What is his age raised to the second power??",
        "Who won the US Open men's tennis final in 2022? What is his age raised to the third power??",
        "Who won the US Open men's tennis final in 2022? What is his age raised to the second power??",
        "Who won the US Open men's tennis final in 2022? What is his age raised to the fourth power??",
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in inputs[:3]:
            futures.append(executor.submit(agent.run, i))
        for future in concurrent.futures.as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
