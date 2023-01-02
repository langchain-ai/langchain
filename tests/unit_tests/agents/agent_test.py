import concurrent.futures
import os
import threading

import langchain
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import get_callback_manager, set_default_callback_manager
from langchain.callbacks.tracers import SharedJsonTracer
from langchain.llms import OpenAI

# get_callback_manager().add_handler(SharedJsonTracer())
# langchain.verbose = True


def main():
    llm = OpenAI(temperature=0, verbose=True)
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
        tools, llm, agent="zero-shot-react-description", verbose=True
    )

    # agent.agent.llm_chain.verbose = True  # TODO: rm this after merging with latest changes

    agent.run(
        "Who won the US Open men's tennis final in 2019? What is his age raised to the second power?"
    )

    # inputs = [
    #     "Who won the US Open men's tennis final in 2019? What is his age raised to the second power??",
    #     "Who won the US Open men's tennis final in 2022? What is his age raised to the third power??",
    #     "Who won the US Open men's tennis final in 2022? What is his age raised to the second power??",
    #     "Who won the US Open men's tennis final in 2022? What is his age raised to the fourth power??",
    # ]
    #
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = []
    #     for i in inputs[:3]:
    #         futures.append(executor.submit(agent.run, i))
    #     for future in concurrent.futures.as_completed(futures):
    #         print(future.result())


if __name__ == "__main__":
    main()
