import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from pydantic import BaseModel, Field


from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.json import (
    SimpleJsonOutputParser,
)
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.utils.json import (
    parse_and_check_json_markdown,
    parse_json_markdown,
    parse_partial_json,
)
from tests.unit_tests.pydantic_utils import _schema

GOOD_JSON = """```json
{
    "foo": "bar"
}
```"""

JSON_WITH_NEW_LINES = """

```json
{
    "foo": "bar"
}
```

"""

JSON_WITH_NEW_LINES_INSIDE = """```json
{

    "foo": "bar"

}
```"""

JSON_WITH_NEW_LINES_EVERYWHERE = """

```json

{

    "foo": "bar"

}

```

"""

TICKS_WITH_NEW_LINES_EVERYWHERE = """

```

{

    "foo": "bar"

}

```

"""

JSON_WITH_MARKDOWN_CODE_BLOCK = """```json
{
    "foo": "```bar```"
}
```"""

JSON_WITH_PART_MARKDOWN_CODE_BLOCK = """
{\"valid_json\": "hey ```print(hello world!)``` hey"}
"""

JSON_WITH_MARKDOWN_CODE_BLOCK_AND_NEWLINES = """```json
{
    "action": "Final Answer",
    "action_input": "```bar\n<div id=\\"1\\" class=\\"value\\">\n\ttext\n</div>```"
}
```"""

JSON_WITH_PYTHON_DICT = """```json
{
    "action": "Final Answer",
    "action_input": {"foo": "bar", "bar": "foo"}
}
```"""

JSON_WITH_ESCAPED_DOUBLE_QUOTES_IN_NESTED_JSON = """```json
{
    "action": "Final Answer",
    "action_input": "{\\"foo\\": \\"bar\\", \\"bar\\": \\"foo\\"}"
}
```"""

NO_TICKS = """{
    "foo": "bar"
}"""

NO_TICKS_WHITE_SPACE = """
{
    "foo": "bar"
}
"""

TEXT_BEFORE = """Thought: I need to use the search tool

Action:
```
{
  "foo": "bar"
}
```"""

TEXT_AFTER = """```
{
  "foo": "bar"
}
```
This should do the trick"""

TEXT_BEFORE_AND_AFTER = """Action: Testing

```
{
  "foo": "bar"
}
```
This should do the trick"""

WITHOUT_END_BRACKET = """Here is a response formatted as schema:

```json
{
  "foo": "bar"


"""

WITH_END_BRACKET = """Here is a response formatted as schema:

```json
{
  "foo": "bar"
}

"""

WITH_END_TICK = """Here is a response formatted as schema:

```json
{
  "foo": "bar"
}
```
"""

WITH_END_TEXT = """Here is a response formatted as schema:

```
{
  "foo": "bar"

```
This should do the trick
"""

TEST_CASES = [
    GOOD_JSON,
    JSON_WITH_NEW_LINES,
    JSON_WITH_NEW_LINES_INSIDE,
    JSON_WITH_NEW_LINES_EVERYWHERE,
    TICKS_WITH_NEW_LINES_EVERYWHERE,
    NO_TICKS,
    NO_TICKS_WHITE_SPACE,
    TEXT_BEFORE,
    TEXT_AFTER,
    TEXT_BEFORE_AND_AFTER,
    WITHOUT_END_BRACKET,
    WITH_END_BRACKET,
    WITH_END_TICK,
    WITH_END_TEXT,
]


@pytest.mark.parametrize("json_string", TEST_CASES)
def test_parse_json(json_string: str) -> None:
    parsed = parse_json_markdown(json_string)
    assert parsed == {"foo": "bar"}


def test_parse_json_with_code_blocks() -> None:
    parsed = parse_json_markdown(JSON_WITH_MARKDOWN_CODE_BLOCK)
    assert parsed == {"foo": "```bar```"}


def test_parse_json_with_part_code_blocks() -> None:
    parsed = parse_json_markdown(JSON_WITH_PART_MARKDOWN_CODE_BLOCK)
    assert parsed == {"valid_json": "hey ```print(hello world!)``` hey"}


def test_parse_json_with_code_blocks_and_newlines() -> None:
    parsed = parse_json_markdown(JSON_WITH_MARKDOWN_CODE_BLOCK_AND_NEWLINES)
    assert parsed == {
        "action": "Final Answer",
        "action_input": '```bar\n<div id="1" class="value">\n\ttext\n</div>```',
    }


def test_parse_non_dict_json_output() -> None:
    text = "```json\n1\n```"
    with pytest.raises(OutputParserException) as exc_info:
        parse_and_check_json_markdown(text, expected_keys=["foo"])

    assert "Expected JSON object (dict)" in str(exc_info.value)


TEST_CASES_ESCAPED_QUOTES = [
    JSON_WITH_ESCAPED_DOUBLE_QUOTES_IN_NESTED_JSON,
]


@pytest.mark.parametrize("json_string", TEST_CASES_ESCAPED_QUOTES)
def test_parse_nested_json_with_escaped_quotes(json_string: str) -> None:
    parsed = parse_json_markdown(json_string)
    assert parsed == {
        "action": "Final Answer",
        "action_input": '{"foo": "bar", "bar": "foo"}',
    }


def test_parse_json_with_python_dict() -> None:
    parsed = parse_json_markdown(JSON_WITH_PYTHON_DICT)
    assert parsed == {
        "action": "Final Answer",
        "action_input": {"foo": "bar", "bar": "foo"},
    }


TEST_CASES_PARTIAL = [
    ('{"foo": "bar", "bar": "foo"}', '{"foo": "bar", "bar": "foo"}'),
    ('{"foo": "bar", "bar": "foo', '{"foo": "bar", "bar": "foo"}'),
    ('{"foo": "bar", "bar": "foo}', '{"foo": "bar", "bar": "foo}"}'),
    ('{"foo": "bar", "bar": "foo[', '{"foo": "bar", "bar": "foo["}'),
    ('{"foo": "bar", "bar": "foo\\"', '{"foo": "bar", "bar": "foo\\""}'),
    ('{"foo": "bar", "bar":', '{"foo": "bar"}'),
    ('{"foo": "bar", "bar"', '{"foo": "bar"}'),
    ('{"foo": "bar", ', '{"foo": "bar"}'),
    ('{"foo":"bar\\', '{"foo": "bar"}'),
]


@pytest.mark.parametrize("json_strings", TEST_CASES_PARTIAL)
def test_parse_partial_json(json_strings: tuple[str, str]) -> None:
    case, expected = json_strings
    parsed = parse_partial_json(case)
    assert parsed == json.loads(expected)


STREAMED_TOKENS = """
{

 "
setup
":
 "
Why
 did
 the
 bears
 start
 a
 band
 called
 Bears
 Bears
 Bears
 ?
"
,
 "
punchline
":
 "
Because
 they
 wanted
 to
 play
 bear
 -y
 good
 music
 !
"
,
 "
audience
":
 [
"
Haha
"
,
 "
So
 funny
"
]

}
""".splitlines()


EXPECTED_STREAMED_JSON = [
    {},
    {"setup": ""},
    {"setup": "Why"},
    {"setup": "Why did"},
    {"setup": "Why did the"},
    {"setup": "Why did the bears"},
    {"setup": "Why did the bears start"},
    {"setup": "Why did the bears start a"},
    {"setup": "Why did the bears start a band"},
    {"setup": "Why did the bears start a band called"},
    {"setup": "Why did the bears start a band called Bears"},
    {"setup": "Why did the bears start a band called Bears Bears"},
    {"setup": "Why did the bears start a band called Bears Bears Bears"},
    {"setup": "Why did the bears start a band called Bears Bears Bears ?"},
    {
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "punchline": "",
    },
    {
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "punchline": "Because",
    },
    {
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "punchline": "Because they",
    },
    {
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "punchline": "Because they wanted",
    },
    {
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "punchline": "Because they wanted to",
    },
    {
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "punchline": "Because they wanted to play",
    },
    {
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "punchline": "Because they wanted to play bear",
    },
    {
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "punchline": "Because they wanted to play bear -y",
    },
    {
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "punchline": "Because they wanted to play bear -y good",
    },
    {
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "punchline": "Because they wanted to play bear -y good music",
    },
    {
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "punchline": "Because they wanted to play bear -y good music !",
    },
    {
        "punchline": "Because they wanted to play bear -y good music !",
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "audience": [],
    },
    {
        "punchline": "Because they wanted to play bear -y good music !",
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "audience": [""],
    },
    {
        "punchline": "Because they wanted to play bear -y good music !",
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "audience": ["Haha"],
    },
    {
        "punchline": "Because they wanted to play bear -y good music !",
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "audience": ["Haha", ""],
    },
    {
        "punchline": "Because they wanted to play bear -y good music !",
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "audience": ["Haha", "So"],
    },
    {
        "punchline": "Because they wanted to play bear -y good music !",
        "setup": "Why did the bears start a band called Bears Bears Bears ?",
        "audience": ["Haha", "So funny"],
    },
]

EXPECTED_STREAMED_JSON_DIFF = [
    [{"op": "replace", "path": "", "value": {}}],
    [{"op": "add", "path": "/setup", "value": ""}],
    [{"op": "replace", "path": "/setup", "value": "Why"}],
    [{"op": "replace", "path": "/setup", "value": "Why did"}],
    [{"op": "replace", "path": "/setup", "value": "Why did the"}],
    [{"op": "replace", "path": "/setup", "value": "Why did the bears"}],
    [{"op": "replace", "path": "/setup", "value": "Why did the bears start"}],
    [{"op": "replace", "path": "/setup", "value": "Why did the bears start a"}],
    [{"op": "replace", "path": "/setup", "value": "Why did the bears start a band"}],
    [
        {
            "op": "replace",
            "path": "/setup",
            "value": "Why did the bears start a band called",
        }
    ],
    [
        {
            "op": "replace",
            "path": "/setup",
            "value": "Why did the bears start a band called Bears",
        }
    ],
    [
        {
            "op": "replace",
            "path": "/setup",
            "value": "Why did the bears start a band called Bears Bears",
        }
    ],
    [
        {
            "op": "replace",
            "path": "/setup",
            "value": "Why did the bears start a band called Bears Bears Bears",
        }
    ],
    [
        {
            "op": "replace",
            "path": "/setup",
            "value": "Why did the bears start a band called Bears Bears Bears ?",
        }
    ],
    [{"op": "add", "path": "/punchline", "value": ""}],
    [{"op": "replace", "path": "/punchline", "value": "Because"}],
    [{"op": "replace", "path": "/punchline", "value": "Because they"}],
    [{"op": "replace", "path": "/punchline", "value": "Because they wanted"}],
    [{"op": "replace", "path": "/punchline", "value": "Because they wanted to"}],
    [{"op": "replace", "path": "/punchline", "value": "Because they wanted to play"}],
    [
        {
            "op": "replace",
            "path": "/punchline",
            "value": "Because they wanted to play bear",
        }
    ],
    [
        {
            "op": "replace",
            "path": "/punchline",
            "value": "Because they wanted to play bear -y",
        }
    ],
    [
        {
            "op": "replace",
            "path": "/punchline",
            "value": "Because they wanted to play bear -y good",
        }
    ],
    [
        {
            "op": "replace",
            "path": "/punchline",
            "value": "Because they wanted to play bear -y good music",
        }
    ],
    [
        {
            "op": "replace",
            "path": "/punchline",
            "value": "Because they wanted to play bear -y good music !",
        }
    ],
    [{"op": "add", "path": "/audience", "value": []}],
    [{"op": "add", "path": "/audience/0", "value": ""}],
    [{"op": "replace", "path": "/audience/0", "value": "Haha"}],
    [{"op": "add", "path": "/audience/1", "value": ""}],
    [{"op": "replace", "path": "/audience/1", "value": "So"}],
    [{"op": "replace", "path": "/audience/1", "value": "So funny"}],
]


def test_partial_text_json_output_parser() -> None:
    def input_iter(_: Any) -> Iterator[str]:
        yield from STREAMED_TOKENS

    chain = input_iter | SimpleJsonOutputParser()

    assert list(chain.stream(None)) == EXPECTED_STREAMED_JSON


def test_partial_text_json_output_parser_diff() -> None:
    def input_iter(_: Any) -> Iterator[str]:
        yield from STREAMED_TOKENS

    chain = input_iter | SimpleJsonOutputParser(diff=True)

    assert list(chain.stream(None)) == EXPECTED_STREAMED_JSON_DIFF


async def test_partial_text_json_output_parser_async() -> None:
    async def input_iter(_: Any) -> AsyncIterator[str]:
        for token in STREAMED_TOKENS:
            yield token

    chain = input_iter | SimpleJsonOutputParser()

    assert [p async for p in chain.astream(None)] == EXPECTED_STREAMED_JSON


async def test_partial_text_json_output_parser_diff_async() -> None:
    async def input_iter(_: Any) -> AsyncIterator[str]:
        for token in STREAMED_TOKENS:
            yield token

    chain = input_iter | SimpleJsonOutputParser(diff=True)

    assert [p async for p in chain.astream(None)] == EXPECTED_STREAMED_JSON_DIFF


def test_raises_error() -> None:
    parser = SimpleJsonOutputParser()
    with pytest.raises(OutputParserException):
        parser.invoke("hi")


# A test fixture for an output which contains
# json within a code block
TOKENS_WITH_JSON_CODE_BLOCK = [
    " France",
    ":",
    "\n\n```",
    "json",
    "\n{",
    "\n ",
    ' "',
    "country",
    "_",
    "name",
    '":',
    ' "',
    "France",
    '",',
    " \n ",
    ' "',
    "population",
    "_",
    "size",
    '":',
    " 67",
    "39",
    "15",
    "82",
    "\n}",
    "\n```",
    "\n\nI",
    " looked",
    " up",
]


def test_partial_text_json_output_parser_with_json_code_block() -> None:
    """Test json parser works correctly when the response contains a json code-block."""

    def input_iter(_: Any) -> Iterator[str]:
        yield from TOKENS_WITH_JSON_CODE_BLOCK

    chain = input_iter | SimpleJsonOutputParser()

    assert list(chain.stream(None)) == [
        {},
        {"country_name": ""},
        {"country_name": "France"},
        {"country_name": "France", "population_size": 67},
        {"country_name": "France", "population_size": 6739},
        {"country_name": "France", "population_size": 673915},
        {"country_name": "France", "population_size": 67391582},
    ]


def test_base_model_schema_consistency() -> None:
    class Joke(BaseModel):
        setup: str
        punchline: str

    initial_joke_schema = dict(_schema(Joke).items())
    SimpleJsonOutputParser(pydantic_object=Joke)
    openai_func = convert_to_openai_function(Joke)
    retrieved_joke_schema = dict(_schema(Joke).items())

    assert initial_joke_schema == retrieved_joke_schema
    assert openai_func.get("name", None) is not None


def test_unicode_handling() -> None:
    """Tests if the JsonOutputParser is able to process unicodes."""

    class Sample(BaseModel):
        title: str = Field(description="科学文章的标题")

    parser = SimpleJsonOutputParser(pydantic_object=Sample)
    format_instructions = parser.get_format_instructions()
    assert "科学文章的标题" in format_instructions, (
        "Unicode characters should not be escaped"
    )
