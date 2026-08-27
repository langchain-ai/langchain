import datetime
import json
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langsmith.schemas import Example

from langchain_core.document_loaders import LangSmithLoader
from langchain_core.document_loaders.langsmith import _get_content_from_inputs
from langchain_core.documents import Document
from langchain_core.tracers._compat import pydantic_to_dict


def test_init() -> None:
    LangSmithLoader(api_key="secret")


def test_init_with_client_and_client_kwargs_raises() -> None:
    client = MagicMock()

    with pytest.raises(ValueError, match="Received both `client` and `client_kwargs`"):
        LangSmithLoader(client=client, api_key="secret")


def test_init_with_client_only() -> None:
    """A bare `client` (no `client_kwargs`) should be accepted."""
    client = MagicMock()

    loader = LangSmithLoader(client=client)

    assert loader._client is client


EXAMPLES = [
    Example(
        inputs={"first": {"second": "foo"}},
        outputs={"res": "a"},
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    ),
    Example(
        inputs={"first": {"second": "bar"}},
        outputs={"res": "b"},
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    ),
    Example(
        inputs={"first": {"second": "baz"}},
        outputs={"res": "c"},
        dataset_id=uuid.uuid4(),
        id=uuid.uuid4(),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    ),
]


@patch("langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES)))
def test_lazy_load() -> None:
    loader = LangSmithLoader(
        api_key="dummy",
        dataset_id="mock",
        content_key="first.second",
        format_content=(lambda x: x.upper()),
    )
    expected = []
    for example in EXAMPLES:
        example_dict = pydantic_to_dict(example)
        metadata = {
            k: v if not v or isinstance(v, dict) else str(v)
            for k, v in example_dict.items()
        }
        expected.append(
            Document(example.inputs["first"]["second"].upper(), metadata=metadata)
            if example.inputs
            else None
        )
    actual = list(loader.lazy_load())
    assert expected == actual


@patch("langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES[:1])))
def test_lazy_load_with_empty_content_key_returns_whole_inputs() -> None:
    """An empty `content_key` (the default) yields the full inputs payload."""
    loader = LangSmithLoader(api_key="dummy", dataset_id="mock")

    docs = list(loader.lazy_load())

    assert len(docs) == 1
    assert docs[0].page_content == json.dumps({"first": {"second": "foo"}}, indent=2)


@patch("langsmith.Client.list_examples", MagicMock(return_value=iter(EXAMPLES[:1])))
def test_lazy_load_with_missing_content_key_raises() -> None:
    loader = LangSmithLoader(
        api_key="dummy",
        dataset_id="mock",
        content_key="first.third",
    )

    with pytest.raises(
        ValueError,
        match=r"Could not resolve content_key 'first\.third': "
        r"missing key 'third' under 'first'",
    ):
        list(loader.lazy_load())


@pytest.mark.parametrize(
    ("inputs", "content_key", "expected"),
    [
        # Empty key path (the default) returns the whole payload.
        ({"first": {"second": "foo"}}, [], {"first": {"second": "foo"}}),
        # Partial path resolves to an intermediate mapping.
        ({"first": {"second": "foo"}}, ["first"], {"second": "foo"}),
        # Full path resolves to a leaf value.
        ({"first": {"second": "foo"}}, ["first", "second"], "foo"),
    ],
)
def test_get_content_from_inputs_resolves_path(
    inputs: Any, content_key: list[str], expected: Any
) -> None:
    assert _get_content_from_inputs(inputs, content_key) == expected


@pytest.mark.parametrize(
    ("inputs", "content_key", "match"),
    [
        # Missing key at the root level reports the "<root>" context.
        (
            {"first": {"second": "foo"}},
            ["missing"],
            r"missing key 'missing' under '<root>'",
        ),
        # Missing key nested under an existing mapping.
        (
            {"first": {"second": "foo"}},
            ["first", "third"],
            r"missing key 'third' under 'first'",
        ),
        # Path traverses past a leaf value that is not a mapping.
        (
            {"first": {"second": "foo"}},
            ["first", "second", "third"],
            r"expected a mapping at 'first\.second', but found str",
        ),
        # The root inputs payload itself is not a mapping.
        (None, ["first"], r"expected a mapping at '<root>', but found NoneType"),
    ],
)
def test_get_content_from_inputs_raises(
    inputs: Any, content_key: list[str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _get_content_from_inputs(inputs, content_key)
