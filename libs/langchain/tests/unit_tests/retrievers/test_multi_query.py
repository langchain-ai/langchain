from typing import List

import pytest as pytest

from langchain.retrievers.multi_query import _unique_documents
from langchain.schema import Document


@pytest.mark.parametrize(
    "documents,expected",
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
def test__unique_documents(documents: List[Document], expected: List[Document]) -> None:
    assert _unique_documents(documents) == expected
