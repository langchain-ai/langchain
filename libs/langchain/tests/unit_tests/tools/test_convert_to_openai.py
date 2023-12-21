import os

from langchain_community.tools.google_serper.tool import GoogleSerperRun
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.tools import Tool, tool

from langchain.chains import LLMMathChain
from langchain.tools import format_tool_to_openai_function
from tests.unit_tests.llms.fake_llm import FakeLLM


def assert_oai_func(oai_func_desc, parameter_name):
    assert "parameters" in oai_func_desc
    assert "properties" in oai_func_desc["parameters"]
    parameter = oai_func_desc["parameters"]["properties"]
    assert len(parameter) == 1
    assert parameter_name in parameter


def test_function_tool() -> None:
    @tool
    def get_word_length(word: str) -> int:
        """Returns the length of a word."""
        return len(word)

    oai_func_desc = format_tool_to_openai_function(get_word_length)
    assert_oai_func(oai_func_desc, "word")


def test_subclass_of_base_tool() -> None:
    """test tools created by subclassing BaseTool"""
    os.environ["SERPER_API_KEY"] = "1" * 40  # mocked key
    serp_tool = GoogleSerperRun(api_wrapper=GoogleSerperAPIWrapper())
    oai_func_desc = format_tool_to_openai_function(serp_tool)
    assert_oai_func(oai_func_desc, "query")


def test_class_of_tool() -> None:
    def get_word_length(word: str) -> int:
        """Returns the length of a word."""
        return len(word)

    t = Tool(
        name="word_length",
        description="Useful for count word length.",
        func=get_word_length,
    )
    oai_func_desc = format_tool_to_openai_function(t)
    assert_oai_func(oai_func_desc, "word")


def test_class_of_tool_with_chain_run() -> None:
    llm = FakeLLM()
    calculator = Tool(
        name="Calculator",
        description="Useful for when you need to answer questions about math.",
        func=LLMMathChain.from_llm(llm=llm).run,
        coroutine=LLMMathChain.from_llm(llm=llm).arun,
    )
    oai_func_desc = format_tool_to_openai_function(calculator)
    assert_oai_func(oai_func_desc, "question")
