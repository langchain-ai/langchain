import tempfile
import urllib.request
from typing import Generator, Iterable

import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores import Vectara
from langchain_community.vectorstores.vectara import (
    MMRConfig,
    RerankConfig,
    SummaryConfig,
    VectaraQueryConfig,
)

#
# For this test to run properly, please setup as follows:
# 1. Create a Vectara account: sign up at https://www.vectara.com/integrations/langchain
# 2. Create a corpus in your Vectara account, with a filter attribute called "test_num".
# 3. Create an API_KEY for this corpus with permissions for query and indexing
# 4. Setup environment variables:
#    VECTARA_API_KEY, VECTARA_CORPUS_ID and VECTARA_CUSTOMER_ID
#

test_prompt_name = "vectara-experimental-summary-ext-2023-12-11-sml"


def get_abbr(s: str) -> str:
    words = s.split(" ")  # Split the string into words
    first_letters = [word[0] for word in words]  # Extract the first letter of each word
    return "".join(first_letters)  # Join the first letters into a single string


@pytest.fixture(scope="function")
def vectara1() -> Iterable[Vectara]:
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


def test_vectara_add_documents(vectara1: Vectara) -> None:
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

    # test with metadata filter (doc level)
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
def vectara2() -> Generator[Vectara, None, None]:
    # download documents to local storage and then upload as files
    # attention paper and deep learning book
    vectara2: Vectara = Vectara()  # type: ignore

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
    """test uploading data from files"""
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

    # Test the old n_sentence_context to ensure it's backward compatible
    output = vectara2.similarity_search(
        "By the commonly adopted machine learning tradition",
        k=1,
        n_sentence_context=1,
        filter="doc.test_num = 2",
    )
    assert "Note the use of" in output[0].page_content


def test_vectara_rag_with_reranking(vectara2: Vectara) -> None:
    """Test Vectara reranking."""

    query_str = "What is a transformer model?"

    # Note: we don't test rerank_multilingual_v1 as it's for Scale only

    # Test MMR
    summary_config = SummaryConfig(
        is_enabled=True,
        max_results=7,
        response_lang="eng",
        prompt_name=test_prompt_name,
    )
    rerank_config = RerankConfig(reranker="mmr", rerank_k=50, mmr_diversity_bias=0.2)
    config = VectaraQueryConfig(
        k=10,
        lambda_val=0.005,
        rerank_config=rerank_config,
        summary_config=summary_config,
    )

    rag1 = vectara2.as_rag(config)
    response1 = rag1.invoke(query_str)

    assert "transformer model" in response1["answer"].lower()

    # Test No reranking
    summary_config = SummaryConfig(
        is_enabled=True,
        max_results=7,
        response_lang="eng",
        prompt_name=test_prompt_name,
    )
    rerank_config = RerankConfig(reranker="None")
    config = VectaraQueryConfig(
        k=10,
        lambda_val=0.005,
        rerank_config=rerank_config,
        summary_config=summary_config,
    )
    rag2 = vectara2.as_rag(config)
    response2 = rag2.invoke(query_str)

    assert "transformer model" in response2["answer"].lower()

    # assert that the page content is different for the top 5 results
    # in each reranking
    n_results = 10
    response1_content = [x[0].page_content for x in response1["context"][:n_results]]
    response2_content = [x[0].page_content for x in response2["context"][:n_results]]
    assert response1_content != response2_content


@pytest.fixture(scope="function")
def vectara3() -> Iterable[Vectara]:
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


def test_vectara_with_langchain_mmr(vectara3: Vectara) -> None:  # type: ignore[no-untyped-def]
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
        "This is why today we're adding a fundamental capability"
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


def test_vectara_mmr(vectara3: Vectara) -> None:  # type: ignore[no-untyped-def]
    # test MMR directly with rerank_config
    summary_config = SummaryConfig(is_enabled=True, max_results=7, response_lang="eng")
    rerank_config = RerankConfig(reranker="mmr", rerank_k=50, mmr_diversity_bias=0.2)
    config = VectaraQueryConfig(
        k=10,
        lambda_val=0.005,
        rerank_config=rerank_config,
        summary_config=summary_config,
    )
    rag = vectara3.as_rag(config)
    output1 = rag.invoke("what is generative AI?")["answer"]
    assert len(output1) > 0

    # test MMR directly with old mmr_config
    summary_config = SummaryConfig(is_enabled=True, max_results=7, response_lang="eng")
    mmr_config = MMRConfig(is_enabled=True, mmr_k=50, diversity_bias=0.2)
    config = VectaraQueryConfig(
        k=10, lambda_val=0.005, mmr_config=mmr_config, summary_config=summary_config
    )
    rag = vectara3.as_rag(config)
    output2 = rag.invoke("what is generative AI?")["answer"]
    assert len(output2) > 0

    # test reranking disabled - RerankConfig
    summary_config = SummaryConfig(is_enabled=True, max_results=7, response_lang="eng")
    rerank_config = RerankConfig(reranker="none")
    config = VectaraQueryConfig(
        k=10,
        lambda_val=0.005,
        rerank_config=rerank_config,
        summary_config=summary_config,
    )
    rag = vectara3.as_rag(config)
    output1 = rag.invoke("what is generative AI?")["answer"]
    assert len(output1) > 0

    # test with reranking disabled - MMRConfig
    summary_config = SummaryConfig(is_enabled=True, max_results=7, response_lang="eng")
    mmr_config = MMRConfig(is_enabled=False, mmr_k=50, diversity_bias=0.2)
    config = VectaraQueryConfig(
        k=10, lambda_val=0.005, mmr_config=mmr_config, summary_config=summary_config
    )
    rag = vectara3.as_rag(config)
    output2 = rag.invoke("what is generative AI?")["answer"]
    assert len(output2) > 0


def test_vectara_with_summary(vectara3) -> None:  # type: ignore[no-untyped-def]
    """Test vectara summary."""
    # test summarization
    num_results = 10
    output1 = vectara3.similarity_search(
        query="what is generative AI?",
        k=num_results,
        summary_config=SummaryConfig(
            is_enabled=True,
            max_results=5,
            response_lang="eng",
            prompt_name=test_prompt_name,
        ),
    )

    assert len(output1) == num_results + 1
    assert len(output1[num_results].page_content) > 0
