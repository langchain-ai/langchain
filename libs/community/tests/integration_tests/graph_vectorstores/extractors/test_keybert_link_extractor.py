import pytest

from langchain_community.graph_vectorstores.extractors import KeybertLinkExtractor
from langchain_community.graph_vectorstores.links import Link

PAGE_1 = """
Supervised learning is the machine learning task of learning a function that
maps an input to an output based on example input-output pairs. It infers a
function from labeled training data consisting of a set of training examples. In
supervised learning, each example is a pair consisting of an input object
(typically a vector) and a desired output value (also called the supervisory
signal). A supervised learning algorithm analyzes the training data and produces
an inferred function, which can be used for mapping new examples. An optimal
scenario will allow for the algorithm to correctly determine the class labels
for unseen instances. This requires the learning algorithm to generalize from
the training data to unseen situations in a 'reasonable' way (see inductive
bias).
"""

PAGE_2 = """
KeyBERT is a minimal and easy-to-use keyword extraction technique that leverages
BERT embeddings to create keywords and keyphrases that are most similar to a
document.
"""


@pytest.mark.requires("keybert")
def test_one_from_keywords() -> None:
    extractor = KeybertLinkExtractor()

    results = extractor.extract_one(PAGE_1)
    assert results == {
        Link.bidir(kind="kw", tag="supervised"),
        Link.bidir(kind="kw", tag="labels"),
        Link.bidir(kind="kw", tag="labeled"),
        Link.bidir(kind="kw", tag="learning"),
        Link.bidir(kind="kw", tag="training"),
    }


@pytest.mark.requires("keybert")
def test_many_from_keyphrases() -> None:
    extractor = KeybertLinkExtractor(
        extract_keywords_kwargs={
            "keyphrase_ngram_range": (1, 2),
        }
    )

    results = list(extractor.extract_many([PAGE_1, PAGE_2]))
    assert results[0] == {
        Link.bidir(kind="kw", tag="supervised"),
        Link.bidir(kind="kw", tag="labeled training"),
        Link.bidir(kind="kw", tag="supervised learning"),
        Link.bidir(kind="kw", tag="examples supervised"),
        Link.bidir(kind="kw", tag="signal supervised"),
    }

    assert results[1] == {
        Link.bidir(kind="kw", tag="keyphrases"),
        Link.bidir(kind="kw", tag="keyword extraction"),
        Link.bidir(kind="kw", tag="keybert"),
        Link.bidir(kind="kw", tag="keywords keyphrases"),
        Link.bidir(kind="kw", tag="keybert minimal"),
    }
