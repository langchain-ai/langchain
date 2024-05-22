import tempfile
import urllib.request

import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores.vectara import SummaryConfig, Vectara

#
# For this test to run properly, please setup as follows:
# 1. Create a Vectara account: sign up at https://www.vectara.com/integrations/langchain
# 2. Create a corpus in your Vectara account, with a filter attribute called "test_num".
# 3. Create an API_KEY for this corpus with permissions for query and indexing
# 4. Setup environment variables:
#    VECTARA_API_KEY, VECTARA_CORPUS_ID and VECTARA_CUSTOMER_ID
#


def get_abbr(s: str) -> str:
    words = s.split(" ")  # Split the string into words
    first_letters = [word[0] for word in words]  # Extract the first letter of each word
    return "".join(first_letters)  # Join the first letters into a single string


@pytest.fixture(scope="function")
def vectara1():  # type: ignore[no-untyped-def]
    # Set up code
    # create a new Vectara instance
    vectara1: Vectara = Vectara()

    # start with some initial texts, added with add_texts
    texts1 = ["grounded generation", "retrieval augmented generation", "data privacy"]
    md = [{"abbr": get_abbr(t)} for t in texts1]
    doc_id1 = vectara1.add_texts(
        texts1,
        metadatas=md,
        doc_metadata={"test_num": "1"},
    )

    # then add some additional documents, now with add_documents
    texts2 = ["large language model", "information retrieval", "question answering"]
    doc_id2 = vectara1.add_documents(
        [Document(page_content=t, metadata={"abbr": get_abbr(t)}) for t in texts2],
        doc_metadata={"test_num": "2"},
    )
    doc_ids = doc_id1 + doc_id2

    yield vectara1

    # Tear down code
    vectara1.delete(doc_ids)


def test_vectara_add_documents(vectara1: Vectara) -> None:  # type: ignore[no-untyped-def]
    """Test add_documents."""

    # test without filter
    output1 = vectara1.similarity_search(
        "large language model",
        k=2,
        n_sentence_before=0,
        n_sentence_after=0,
    )
    assert len(output1) == 2
    assert output1[0].page_content == "large language model"
    assert output1[0].metadata["abbr"] == "llm"
    assert output1[1].page_content == "retrieval augmented generation"
    assert output1[1].metadata["abbr"] == "rag"

    # test with metadata filter (doc level)
    # since the query does not match test_num=1 directly we get "LLM" as the result
    output2 = vectara1.similarity_search(
        "large language model",
        k=1,
        n_sentence_before=0,
        n_sentence_after=0,
        filter="doc.test_num = 1",
    )
    assert len(output2) == 1
    assert output2[0].page_content == "retrieval augmented generation"
    assert output2[0].metadata["abbr"] == "rag"

    # test without filter but with similarity score
    # this is similar to the first test, but given the score threshold
    # we only get one result
    output3 = vectara1.similarity_search_with_score(
        "large language model",
        k=2,
        score_threshold=0.5,
        n_sentence_before=0,
        n_sentence_after=0,
    )
    assert len(output3) == 2
    assert output3[0][0].page_content == "large language model"
    assert output3[0][0].metadata["abbr"] == "llm"


@pytest.fixture(scope="function")
def vectara2():  # type: ignore[no-untyped-def]
    # download documents to local storage and then upload as files
    # attention paper and deep learning book
    vectara2: Vectara = Vectara()

    urls = [
        (
            "https://papers.nips.cc/paper_files/paper/2017/"
            "file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
        ),
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

    doc_ids = vectara2.add_files(
        files_list=files_list,
        metadatas=[{"url": url, "test_num": "2"} for url in urls],
    )

    yield vectara2

    # Tear down code
    vectara2.delete(doc_ids)


def test_vectara_from_files(vectara2: Vectara) -> None:
    """Test end to end construction and search."""
    # finally do a similarity search to see if all works okay
    output = vectara2.similarity_search(
        "By the commonly adopted machine learning tradition",
        k=1,
        n_sentence_before=0,
        n_sentence_after=0,
        filter="doc.test_num = 2",
    )
    assert (
        "By the commonly adopted machine learning tradition" in output[0].page_content
    )

    # another similarity search, this time with n_sentences_before/after = 1
    output = vectara2.similarity_search(
        "By the commonly adopted machine learning tradition",
        k=1,
        n_sentence_before=1,
        n_sentence_after=1,
        filter="doc.test_num = 2",
    )
    assert "Note the use of" in output[0].page_content


@pytest.fixture(scope="function")
def vectara3():  # type: ignore[no-untyped-def]
    # Set up code
    vectara3: Vectara = Vectara()

    # start with some initial texts, added with add_texts
    texts = [
        """
        The way Grounded Generation with Vectara works is we only use valid responses 
        from your data relative to the search query. 
        This dramatically reduces hallucinations in Vectara's responses. 
        You can try it out on your own on our newly launched AskNews demo to experience 
        Grounded Generation, or register an account to ground generative summaries on 
        your own data.
        """,
        """
        Generative AI promises to revolutionize how you can benefit from your data, 
        but you need it to provide dependable information without the risk of data 
        leakage. This is why today we're adding a fundamental capability to our 
        platform to make generative AI safer to use. It enables you to ask your 
        data questions and get reliable, accurate answers by retrieving and 
        summarizing only the relevant information. We call it “Grounded Generation”. 
        """,
        """
        We are incredibly excited to share another feature with this launch: 
        Hybrid Search! Neural LLM systems are excellent at understanding the context 
        and meaning of end-user queries, but they can still underperform when matching 
        exact product SKUs, unusual names of people or companies, barcodes, and other 
        text which identifies entities rather than conveying semantics. We're bridging 
        this gap by introducing a lexical configuration that matches exact keywords, 
        supports Boolean operators, and executes phrase searches, and incorporates 
        the results into our neural search results.
        """,
    ]

    doc_ids = []
    for text in texts:
        ids = vectara3.add_documents([Document(page_content=text, metadata={})])
        doc_ids.extend(ids)

    yield vectara3

    # Tear down code
    vectara3.delete(doc_ids)


def test_vectara_mmr(vectara3: Vectara) -> None:  # type: ignore[no-untyped-def]
    # test max marginal relevance
    output1 = vectara3.max_marginal_relevance_search(
        "generative AI",
        k=2,
        fetch_k=6,
        lambda_mult=1.0,  # no diversity bias
        n_sentence_before=0,
        n_sentence_after=0,
    )
    assert len(output1) == 2
    assert (
        "This is why today we're adding a fundamental capability to our"
        in output1[1].page_content
    )

    output2 = vectara3.max_marginal_relevance_search(
        "generative AI",
        k=2,
        fetch_k=6,
        lambda_mult=0.0,  # only diversity bias
        n_sentence_before=0,
        n_sentence_after=0,
    )
    assert len(output2) == 2
    assert (
        "Neural LLM systems are excellent at understanding the context"
        in output2[1].page_content
    )


def test_vectara_with_summary(vectara3) -> None:  # type: ignore[no-untyped-def]
    """Test vectara summary."""
    # test summarization
    num_results = 10
    output1 = vectara3.similarity_search(
        query="what is generative AI?",
        k=num_results,
        summary_config=SummaryConfig(is_enabled=True, max_results=5),
    )

    assert len(output1) == num_results + 1
    assert len(output1[num_results].page_content) > 0
