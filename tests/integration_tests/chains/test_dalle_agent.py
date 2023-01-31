from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

def test_call() -> None:
    llm = OpenAI(temperature=0.9)
    tools = load_tools(['dalle-image-generator'])

    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    output = agent.run("Create an image of a volcano island")
    assert output is not None