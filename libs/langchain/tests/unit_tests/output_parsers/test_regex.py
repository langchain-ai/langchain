from typing import Dict

from langchain.output_parsers.regex import RegexParser

# NOTE: The almost same constant variables in ./test_combining_parser.py
DEF_EXPECTED_RESULT = {
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


def test_regex_parser_parse() -> None:
    """Test regex parser parse."""
    parser = RegexParser(
        regex=r"Confidence: (A|B|C), Explanation: (.*)",
        output_keys=["confidence", "explanation"],
        default_output_key="noConfidence",
    )
    assert DEF_EXPECTED_RESULT == parser.parse(DEF_README)


def test_regex_parser_output_type() -> None:
    """Test regex parser output type is Dict[str, str]."""
    parser = RegexParser(
        regex=r"Confidence: (A|B|C), Explanation: (.*)",
        output_keys=["confidence", "explanation"],
        default_output_key="noConfidence",
    )
    assert parser.OutputType is Dict[str, str]
