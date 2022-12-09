"""Load tools."""
from langchain.agents.tools import Tool
from typing import List
from langchain.python import PythonREPL
from langchain.serpapi import SerpAPIWrapper
from langchain.requests import RequestsWrapper
from langchain.utilities.bash import BashProcess


TOOLS = {
    "python_repl": Tool(
        "Python REPL",
        PythonREPL().run,
        "A Python shell. Use this to execute python commands. Input should be a valid python command. If you expect output it should be printed out."
    ),
    # "serpapi": Tool(
    #     "Search",
    #     SerpAPIWrapper().run,
    #     "A search engine. Useful for when you need to answer questions about current events. Input should be a search query."
    # ),
    "requests": Tool(
        "Requests",
        RequestsWrapper().run,
        "A portal to the internet. Use this when you need to get specific content from a site. Input should be a specific url, and the output will be all the text on that page."
    ),
    "terminal": Tool(
        "Terminal",
        BashProcess().run,
        "Executes commands in a terminal. Input should be valid commands, and the output will be any output from running that command."
    ),
}

def load_tools(tool_names: List[str]) -> List[Tool]:
    return [TOOLS[name] for name in tool_names]
