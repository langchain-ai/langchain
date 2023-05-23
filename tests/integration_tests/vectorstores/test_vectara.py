from langchain.docstore.document import Document
from langchain.vectorstores.vectara import Vectara
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def get_abbr(s: str) -> str:
    words = s.split(" ")  # Split the string into words
    first_letters = [word[0] for word in words]  # Extract the first letter of each word
    return "".join(first_letters)  # Join the first letters into a single string


def test_vectara_add_documents() -> None:
    """Test end to end construction and search."""

    # start with some initial documents
    texts = ["grounded generation", "retrieval augmented generation", "data privacy"]
    docsearch: Vectara = Vectara.from_texts(
        texts,
        embedding=FakeEmbeddings(),
        metadatas=[{"abbr": "gg"}, {"abbr": "rag"}, {"abbr": "dp"}],
    )

    # then add some additional documents
    new_texts = ["large language model", "information retrieval", "question answering"]
    docsearch.add_documents(
        [Document(page_content=t, metadata={"abbr": get_abbr(t)}) for t in new_texts]
    )

    # finally do a similarity search to see if all works okay
    output = docsearch.similarity_search("large language model", k=2)
    assert output[0].page_content == "large language model"
    assert output[0].metadata == {"abbr": "llm"}
    assert output[1].page_content == "information retrieval"
    assert output[1].metadata == {"abbr": "ir"}
