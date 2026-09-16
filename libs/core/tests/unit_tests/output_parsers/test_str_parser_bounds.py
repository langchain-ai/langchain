\"\"\"
Unit tests for StrOutputParser input variations, AIMessage extraction, and empty string bounds in langchain_core.output_parsers.
\"\"\"
import pytest
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

def test_str_output_parser_from_ai_message():
    parser = StrOutputParser()
    msg = AIMessage(content="Generated answer.")
    assert parser.invoke(msg) == "Generated answer."

def test_str_output_parser_from_plain_string():
    parser = StrOutputParser()
    assert parser.invoke("Raw text response.") == "Raw text response."

def test_str_output_parser_empty_content():
    parser = StrOutputParser()
    msg = AIMessage(content="")
    assert parser.invoke(msg) == ""