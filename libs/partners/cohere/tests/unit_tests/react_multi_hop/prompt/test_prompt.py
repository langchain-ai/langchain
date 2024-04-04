from typing import Any

import pytest
from langchain_core.messages import SystemMessage

from langchain_cohere.react_multi_hop.prompt import render_observations


def test_render_observation_has_correct_indexes() -> None:
    index = 13
    observations = ["foo", "bar"]
    expected_index = 15

    _, actual = render_observations(observations=observations, index=index)

    assert expected_index == actual


document_template = """Document: {index}
{fields}"""


@pytest.mark.parametrize(
    "observation,expected_content",
    [
        pytest.param(
            "foo", document_template.format(index=0, fields="Output: foo"), id="string"
        ),
        pytest.param(
            {"foo": "bar"},
            document_template.format(index=0, fields="Foo: bar"),
            id="dictionary",
        ),
        pytest.param(
            {"url": "foo"},
            document_template.format(index=0, fields="URL: foo"),
            id="dictionary with url",
        ),
        pytest.param(
            {"foo": "bar", "baz": "foobar"},
            document_template.format(index=0, fields="Foo: bar\nBaz: foobar"),
            id="dictionary with multiple keys",
        ),
        pytest.param(
            ["foo", "bar"],
            "\n\n".join(
                [
                    document_template.format(index=0, fields="Output: foo"),
                    document_template.format(index=1, fields="Output: bar"),
                ]
            ),
            id="list of strings",
        ),
        pytest.param(
            [{"foo": "bar"}, {"baz": "foobar"}],
            "\n\n".join(
                [
                    document_template.format(index=0, fields="Foo: bar"),
                    document_template.format(index=1, fields="Baz: foobar"),
                ]
            ),
            id="list of dictionaries",
        ),
        pytest.param(
            2,
            document_template.format(index=0, fields="Output: 2"),
            id="int",
        ),
        pytest.param(
            [2],
            document_template.format(index=0, fields="Output: 2"),
            id="list of int",
        ),
    ],
)
def test_render_observation_has_correct_content(
    observation: Any, expected_content: str
) -> None:
    actual, _ = render_observations(observations=observation, index=0)
    expected_content = f"<results>\n{expected_content}\n</results>"

    assert SystemMessage(content=expected_content) == actual
