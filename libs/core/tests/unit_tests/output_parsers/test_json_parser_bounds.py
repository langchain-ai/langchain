\"\"\"
Unit tests for JsonOutputParser markdown block parsing and JSON decoding bounds in langchain_core.output_parsers.
\"\"\"
import pytest
from langchain_core.output_parsers import JsonOutputParser

def test_json_output_parser_clean_json():
    parser = JsonOutputParser()
    raw = '{\"name\": \"LangChain\", \"version\": 1}'
    result = parser.invoke(raw)
    assert result == {\"name\": \"LangChain\", \"version\": 1}

def test_json_output_parser_markdown_code_block():
    parser = JsonOutputParser()
    raw = '`json\n{\"status\": \"active\", \"code\": 200}\n`'
    result = parser.invoke(raw)
    assert result == {\"status\": \"active\", \"code\": 200}

def test_json_output_parser_invalid_json_handling():
    parser = JsonOutputParser()
    with pytest.raises(Exception):
        parser.invoke('Not a valid JSON string')