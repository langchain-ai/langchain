from typing import List, Mapping

import pytest

from langchain_cohere import CohereCitation
from langchain_cohere.react_multi_hop.parsing import parse_citations

DOCUMENTS = [{"foo": "bar"}, {"baz": "foobar"}]


@pytest.mark.parametrize(
    "text,documents,expected_generation,expected_citations",
    [
        pytest.param(
            "no citations",
            DOCUMENTS,
            "no citations",
            [],
            id="no citations",
        ),
        pytest.param(
            "with <co: 0>one citation</co: 0>.",
            DOCUMENTS,
            "with one citation.",
            [
                CohereCitation(
                    start=5, end=17, text="one citation", documents=[DOCUMENTS[0]]
                )
            ],
            id="one citation (normal)",
        ),
        pytest.param(
            "with <co: 0,1>two documents</co: 0,1>.",
            DOCUMENTS,
            "with two documents.",
            [
                CohereCitation(
                    start=5,
                    end=18,
                    text="two documents",
                    documents=[DOCUMENTS[0], DOCUMENTS[1]],
                )
            ],
            id="two cited documents (normal)",
        ),
        pytest.param(
            "with <co: 0>two</co: 0> <co: 1>citations</co: 1>.",
            DOCUMENTS,
            "with two citations.",
            [
                CohereCitation(start=5, end=8, text="two", documents=[DOCUMENTS[0]]),
                CohereCitation(
                    start=9, end=18, text="citations", documents=[DOCUMENTS[1]]
                ),
            ],
            id="more than one citation (normal)",
        ),
        pytest.param(
            "with <co: 2>incorrect citation</co: 2>.",
            DOCUMENTS,
            "with incorrect citation.",
            [
                CohereCitation(
                    start=5,
                    end=23,
                    text="incorrect citation",
                    documents=[],  # note no documents.
                )
            ],
            id="cited document doesn't exist (abnormal)",
        ),
    ],
)
def test_parse_citations(
    text: str,
    documents: List[Mapping],
    expected_generation: str,
    expected_citations: List[CohereCitation],
) -> None:
    actual_generation, actual_citations = parse_citations(
        grounded_answer=text, documents=documents
    )
    assert expected_generation == actual_generation
    assert expected_citations == actual_citations
    for citation in actual_citations:
        assert text[citation.start : citation.end]
