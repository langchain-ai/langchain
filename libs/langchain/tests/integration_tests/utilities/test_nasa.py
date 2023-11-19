"""Integration test for NASA API Wrapper."""
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.utilities.nasa import NasaAPIWrapper
from langchain.agents.agent_toolkits.nasa.toolkit import NasaToolkit
from langchain.llms import OpenAI
import os

llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
nasa = NasaAPIWrapper()
toolkit = NasaToolkit.from_nasa_api_wrapper(nasa)
agent = initialize_agent(
    toolkit.get_tools(),
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
agent.run("Find me an old picture of Saturn")

# testing purposes:
# from langchain.agents import AgentType
# from langchain.agents import initialize_agent
# from langchain.agents.agent_toolkits.nasa.toolkit import NasaToolkit
# from langchain.llms import OpenAI
# from langchain.utilities.nasa import NasaAPIWrapper
# import os

# def tool_test() -> None:
#     llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
#     nasa = NasaAPIWrapper()
#     toolkit = NasaToolkit.from_nasa_api_wrapper(nasa)
#     agent = initialize_agent(
#         toolkit.get_tools(),
#         llm,
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         verbose=True
#     )
#     output = agent.run("Find me an old picture of Saturn")
#     assert output is not None


# # for 'output' and 'assert' statements, reference PROMPT.py examples
# def test_media_search() -> None:
#     """Test for NASA Image and Video Library media search"""
#     nasa = NasaAPIWrapper()
#     output = nasa.run()
#     assert "" in output

# def test_get_media_metadata_manifest() -> None:
#     """Test for retrieving media metadata manifest from NASA Image and Video Library"""
#     nasa = NasaAPIWrapper()
#     output = nasa.run()
#     assert "" in output

# def test_get_media_metadata_location() -> None:
#     """Test for retrieving media metadata location from NASA Image and Video Library"""
#     nasa = NasaAPIWrapper()
#     output = nasa.run()
#     assert "" in output

# def test_get_video_captions_location() -> None:
#     """Test for retrieving video captions location from NASA Image and Video Library"""
#     nasa = NasaAPIWrapper()
#     output = nasa.run()
#     assert "" in output