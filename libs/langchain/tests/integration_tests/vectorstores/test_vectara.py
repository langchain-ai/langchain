import tempfile
import urllib.request

from langchain.docstore.document import Document
from langchain.vectorstores.vectara import Vectara
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

# For this test to run properly, please setup as follows
# 1. Create a corpus in Vectara, with a filter attribute called "test_num".
# 2. Create an API_KEY for this corpus with permissions for query and indexing
# 3. Setup environment variables:
#    VECTARA_API_KEY, VECTARA_CORPUS_ID and VECTARA_CUSTOMER_ID


def get_abbr(s: str) -> str:
    words = s.split(" ")  # Split the string into words
    first_letters = [word[0] for word in words]  # Extract the first letter of each word
    return "".join(first_letters)  # Join the first letters into a single string


def test_vectara_add_documents() -> None:
    """Test end to end construction and search."""

    # start with some initial texts
    texts = ["grounded generation", "retrieval augmented generation", "data privacy"]
    docsearch: Vectara = Vectara.from_texts(
        texts,
        embedding=FakeEmbeddings(),
        metadatas=[
            {"abbr": "gg", "test_num": "1"},
            {"abbr": "rag", "test_num": "1"},
            {"abbr": "dp", "test_num": "1"},
        ],
        doc_metadata={"test_num": "1"},
    )

    # then add some additional documents
    new_texts = ["large language model", "information retrieval", "question answering"]
    docsearch.add_documents(
        [Document(page_content=t, metadata={"abbr": get_abbr(t)}) for t in new_texts],
        doc_metadata={"test_num": "1"},
    )

    # finally do a similarity search to see if all works okay
    output = docsearch.similarity_search(
        "large language model",
        k=2,
        n_sentence_context=0,
        filter="doc.test_num = 1",
    )
    assert output[0].page_content == "large language model"
    assert output[0].metadata == {"abbr": "llm"}
    assert output[1].page_content == "information retrieval"
    assert output[1].metadata == {"abbr": "ir"}


def test_vectara_from_files() -> None:
    """Test end to end construction and search."""

    # download documents to local storage and then upload as files
    # attention paper and deep learning book
    urls = [
        ("https://arxiv.org/pdf/1706.03762.pdf"),
        (
            "https://www.microsoft.com/en-us/research/wp-content/uploads/"
            "2016/02/Final-DengYu-NOW-Book-DeepLearn2013-ForLecturesJuly2.docx"
        ),
    ]

    files_list = []
    for url in urls:
        name = tempfile.NamedTemporaryFile().name
        urllib.request.urlretrieve(url, name)
        files_list.append(name)

    docsearch: Vectara = Vectara.from_files(
        files=files_list,
        embedding=FakeEmbeddings(),
        metadatas=[{"url": url, "test_num": "2"} for url in urls],
    )

    # finally do a similarity search to see if all works okay
    output = docsearch.similarity_search(
        "By the commonly adopted machine learning tradition",
        k=1,
        n_sentence_context=0,
        filter="doc.test_num = 2",
    )
    print(output)
    assert output[0].page_content == (
        "By the commonly adopted machine learning tradition "
        "(e.g., Chapter 28 in Murphy, 2012; Deng and Li, 2013), it may be natural "
        "to just classify deep learning techniques into deep discriminative models "
        "(e.g., DNNs) and deep probabilistic generative models (e.g., DBN, Deep "
        "Boltzmann Machine (DBM))."
    )
