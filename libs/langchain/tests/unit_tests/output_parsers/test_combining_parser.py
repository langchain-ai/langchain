"""Test in memory docstore."""

from typing import Any

from langchain_classic.output_parsers.combining import CombiningOutputParser
from langchain_classic.output_parsers.regex import RegexParser
from langchain_classic.output_parsers.structured import (
    ResponseSchema,
    StructuredOutputParser,
)

DEF_EXPECTED_RESULT = {
    "answer": "Paris",
    "source": "https://en.wikipedia.org/wiki/France",
    "confidence": "A",
    "explanation": "Paris is the capital of France according to Wikipedia.",
}

DEF_README = """```json
{
    "answer": "Paris",
    "source": "https://en.wikipedia.org/wiki/France"
}
```

//Confidence: A, Explanation: Paris is the capital of France according to Wikipedia."""


def test_combining_dict_result() -> None:
    """Test combining result."""
    parsers = [
        StructuredOutputParser(
            response_schemas=[
                ResponseSchema(
                    name="answer",
                    description="answer to the user's question",
                ),
                ResponseSchema(
                    name="source",
                    description="source used to answer the user's question",
                ),
            ],
        ),
        RegexParser(
            regex=r"Confidence: (A|B|C), Explanation: (.*)",
            output_keys=["confidence", "explanation"],
            default_output_key="noConfidence",
        ),
    ]
    combining_parser = CombiningOutputParser(parsers=parsers)
    result_dict = combining_parser.parse(DEF_README)
    assert result_dict == DEF_EXPECTED_RESULT


def test_combining_output_parser_output_type() -> None:
    """Test combining output parser output type is Dict[str, Any]."""
    parsers = [
        StructuredOutputParser(
            response_schemas=[
                ResponseSchema(
                    name="answer",
                    description="answer to the user's question",
                ),
                ResponseSchema(
                    name="source",
                    description="source used to answer the user's question",
                ),
            ],
        ),
        RegexParser(
            regex=r"Confidence: (A|B|C), Explanation: (.*)",
            output_keys=["confidence", "explanation"],
            default_output_key="noConfidence",
        ),
    ]
    combining_parser = CombiningOutputParser(parsers=parsers)
    assert combining_parser.OutputType == dict[str, Any]
