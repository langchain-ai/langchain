from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.messages import AIMessageChunk
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser

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

JSON_WITH_MARKDOWN_CODE_BLOCK_AND_NEWLINES = """```json
{
    "action": "Final Answer",
    "action_input": "```bar\n<div id="1" class=\"value\">\n\ttext\n</div>```"
}
```"""

JSON_WITH_UNESCAPED_QUOTES_IN_NESTED_JSON = """```json
{
    "action": "Final Answer",
    "action_input": "{"foo": "bar", "bar": "foo"}"
}
```"""

JSON_WITH_ESCAPED_QUOTES_IN_NESTED_JSON = """```json
{
    "action": "Final Answer",
    "action_input": "{\"foo\": \"bar\", \"bar\": \"foo\"}"
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
]


TEST_CASES_ESCAPED_QUOTES = [
    JSON_WITH_UNESCAPED_QUOTES_IN_NESTED_JSON,
    JSON_WITH_ESCAPED_QUOTES_IN_NESTED_JSON,
    JSON_WITH_ESCAPED_DOUBLE_QUOTES_IN_NESTED_JSON,
]


TEST_CASES_PARTIAL = [
    ('{"foo": "bar", "bar": "foo"}', '{"foo": "bar", "bar": "foo"}'),
    ('{"foo": "bar", "bar": "foo', '{"foo": "bar", "bar": "foo"}'),
    ('{"foo": "bar", "bar": "foo}', '{"foo": "bar", "bar": "foo}"}'),
    ('{"foo": "bar", "bar": "foo[', '{"foo": "bar", "bar": "foo["}'),
    ('{"foo": "bar", "bar": "foo\\"', '{"foo": "bar", "bar": "foo\\""}'),
]


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


def test_partial_functions_json_output_parser() -> None:
    def input_iter(_: Any) -> Iterator[AIMessageChunk]:
        for token in STREAMED_TOKENS:
            yield AIMessageChunk(
                content="", additional_kwargs={"function_call": {"arguments": token}}
            )

    chain = input_iter | JsonOutputFunctionsParser()

    assert list(chain.stream(None)) == EXPECTED_STREAMED_JSON


def test_partial_functions_json_output_parser_diff() -> None:
    def input_iter(_: Any) -> Iterator[AIMessageChunk]:
        for token in STREAMED_TOKENS:
            yield AIMessageChunk(
                content="", additional_kwargs={"function_call": {"arguments": token}}
            )

    chain = input_iter | JsonOutputFunctionsParser(diff=True)

    assert list(chain.stream(None)) == EXPECTED_STREAMED_JSON_DIFF


async def test_partial_functions_json_output_parser_async() -> None:
    async def input_iter(_: Any) -> AsyncIterator[AIMessageChunk]:
        for token in STREAMED_TOKENS:
            yield AIMessageChunk(
                content="", additional_kwargs={"function_call": {"arguments": token}}
            )

    chain = input_iter | JsonOutputFunctionsParser()

    assert [p async for p in chain.astream(None)] == EXPECTED_STREAMED_JSON


async def test_partial_functions_json_output_parser_diff_async() -> None:
    async def input_iter(_: Any) -> AsyncIterator[AIMessageChunk]:
        for token in STREAMED_TOKENS:
            yield AIMessageChunk(
                content="", additional_kwargs={"function_call": {"arguments": token}}
            )

    chain = input_iter | JsonOutputFunctionsParser(diff=True)

    assert [p async for p in chain.astream(None)] == EXPECTED_STREAMED_JSON_DIFF
