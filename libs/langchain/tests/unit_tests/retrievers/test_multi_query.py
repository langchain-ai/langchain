import pytest
from langchain_core.documents import Document

from langchain.retrievers.multi_query import LineListOutputParser, _unique_documents


@pytest.mark.parametrize(
    ("documents", "expected"),
    [
        ([], []),
        ([Document(page_content="foo")], [Document(page_content="foo")]),
        ([Document(page_content="foo")] * 2, [Document(page_content="foo")]),
        (
            [Document(page_content="foo", metadata={"bar": "baz"})] * 2,
            [Document(page_content="foo", metadata={"bar": "baz"})],
        ),
        (
            [Document(page_content="foo", metadata={"bar": [1, 2]})] * 2,
            [Document(page_content="foo", metadata={"bar": [1, 2]})],
        ),
        (
            [Document(page_content="foo", metadata={"bar": {1, 2}})] * 2,
            [Document(page_content="foo", metadata={"bar": {1, 2}})],
        ),
        (
            [
                Document(page_content="foo", metadata={"bar": [1, 2]}),
                Document(page_content="foo", metadata={"bar": [2, 1]}),
            ],
            [
                Document(page_content="foo", metadata={"bar": [1, 2]}),
                Document(page_content="foo", metadata={"bar": [2, 1]}),
            ],
        ),
    ],
)
def test__unique_documents(documents: list[Document], expected: list[Document]) -> None:
    assert _unique_documents(documents) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("foo\nbar\nbaz", ["foo", "bar", "baz"]),
        ("foo\nbar\nbaz\n", ["foo", "bar", "baz"]),
        ("foo\n\nbar", ["foo", "bar"]),
    ],
)
def test_line_list_output_parser(text: str, expected: list[str]) -> None:
    parser = LineListOutputParser()
    assert parser.parse(text) == expected
