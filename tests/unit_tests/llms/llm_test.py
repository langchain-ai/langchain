import langchain
from langchain.agents import Tool, initialize_agent, load_tools
from langchain.llms import OpenAI
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.tracers import LangChainTracer


if __name__ == "__main__":
    import time

    tracer = LangChainTracer()
    tracer.load_default_session()
    manager = CallbackManager([StdOutCallbackHandler(), tracer])
    llm = OpenAI(temperature=0, callback_manager=manager)
    tools = load_tools(["llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True,
        callback_manager=manager
    )
    agent.run("What is 2 raised to .123243 power?")
