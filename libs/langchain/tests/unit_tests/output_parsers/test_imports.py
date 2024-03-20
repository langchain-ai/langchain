import pytest

from langchain import output_parsers
from tests.unit_tests import assert_all_importable

EXPECTED_ALL = [
    "BooleanOutputParser",
    "CombiningOutputParser",
    "CommaSeparatedListOutputParser",
    "DatetimeOutputParser",
    "EnumOutputParser",
    "ListOutputParser",
    "MarkdownListOutputParser",
    "NumberedListOutputParser",
    "OutputFixingParser",
    "PandasDataFrameOutputParser",
    "PydanticOutputParser",
    "RegexDictParser",
    "RegexParser",
    "ResponseSchema",
    "RetryOutputParser",
    "RetryWithErrorOutputParser",
    "StructuredOutputParser",
    "XMLOutputParser",
    "JsonOutputToolsParser",
    "PydanticToolsParser",
    "JsonOutputKeyToolsParser",
    "YamlOutputParser",
]
EXPECTED_DEPRECATED_IMPORTS = [
    "GuardrailsOutputParser",
]


def test_all_imports() -> None:
    assert set(output_parsers.__all__) == set(EXPECTED_ALL)
    assert_all_importable(output_parsers)


def test_deprecated_imports() -> None:
    for import_ in EXPECTED_DEPRECATED_IMPORTS:
        with pytest.raises(ImportError) as e:
            getattr(output_parsers, import_)
            assert "langchain_community" in e, f"{import_=} didn't error"
    with pytest.raises(AttributeError):
        getattr(output_parsers, "foo")
