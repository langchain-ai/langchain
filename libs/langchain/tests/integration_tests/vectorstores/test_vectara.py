import tempfile
import urllib.request

from langchain.docstore.document import Document
from langchain.vectorstores.vectara import Vectara
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

#
# For this test to run properly, please setup as follows:
# 1. Create a Vectara account: sign up at https://console.vectara.com/signup
# 2. Create a corpus in your Vectara account, with a filter attribute called "test_num".
# 3. Create an API_KEY for this corpus with permissions for query and indexing
# 4. Setup environment variables:
#    VECTARA_API_KEY, VECTARA_CORPUS_ID and VECTARA_CUSTOMER_ID
#


def get_abbr(s: str) -> str:
    words = s.split(" ")  # Split the string into words
    first_letters = [word[0] for word in words]  # Extract the first letter of each word
    return "".join(first_letters)  # Join the first letters into a single string


def test_vectara_add_documents(setup_for_add_documents) -> None:
    """Test end to end construction and search."""

    # create a new Vectara instance
    docsearch: Vectara = Vectara()

    # start with some initial texts, added with add_texts
    texts1 = ["grounded generation", "retrieval augmented generation", "data privacy"]
    md = [{"abbr": get_abbr(t)} for t in texts1]
    doc_id1 = docsearch.add_texts(
        texts1,
        metadatas=md,
        doc_metadata={"test_num": "1"},
    )

    # then add some additional documents, now with add_documents
    texts2 = ["large language model", "information retrieval", "question answering"]
    doc_id2 = docsearch.add_documents(
        [Document(page_content=t, metadata={"abbr": get_abbr(t)}) for t in texts2],
        doc_metadata={"test_num": "2"},
    )
    doc_ids = doc_id1 + doc_id2

    # test without filter
    output1 = docsearch.similarity_search(
        "large language model",
        k=2,
        n_sentence_context=0,
        lambda_val=0,
    )
    assert len(output1) == 2
    assert output1[0].page_content == "large language model"
    assert output1[0].metadata["abbr"] == "llm"
    assert output1[1].page_content == "grounded generation"
    assert output1[1].metadata["abbr"] == "gg"

    # test with metadata filter (doc level)
    # since the query does not match test_num=1 directly we get "RAG" as the result
    output2 = docsearch.similarity_search(
        "large language model",
        k=1,
        n_sentence_context=0,
        filter="doc.test_num = 1",
    )
    assert len(output2) == 1
    assert output2[0].page_content == "grounded generation"
    assert output2[0].metadata["abbr"] == "gg"

    # test without filter but with similarity score
    # this is similar to the first test, but given the score threshold
    # we only get one result
    output3 = docsearch.similarity_search_with_score(
        "large language model",
        k=2,
        score_threshold=0.6,
        n_sentence_context=0,
    )
    assert len(output3) == 1
    assert output3[0][0].page_content == "large language model"
    assert output3[0][0].metadata["abbr"] == "llm"

    for doc_id in doc_ids:
        docsearch._delete_doc(doc_id)


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

    docsearch: Vectara = Vectara()
    doc_ids = docsearch.add_files(
        files_list=files_list,
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
    assert output[0].page_content == (
        "By the commonly adopted machine learning tradition "
        "(e.g., Chapter 28 in Murphy, 2012; Deng and Li, 2013), it may be natural "
        "to just classify deep learning techniques into deep discriminative models "
        "(e.g., DNNs) and deep probabilistic generative models (e.g., DBN, Deep "
        "Boltzmann Machine (DBM))."
    )

    # finally do a similarity search to see if all works okay
    output = docsearch.similarity_search(
        "By the commonly adopted machine learning tradition",
        k=1,
        n_sentence_context=1,
        filter="doc.test_num = 2",
    )
    assert output[0].page_content == (
        """\
Note the use of “hybrid” in 3) above is different from that used sometimes in the literature, \
which for example refers to the hybrid systems for speech recognition feeding the output probabilities of a neural network into an HMM \
(Bengio et al., 1991; Bourlard and Morgan, 1993; Morgan, 2012). \
By the commonly adopted machine learning tradition (e.g., Chapter 28 in Murphy, 2012; Deng and Li, 2013), \
it may be natural to just classify deep learning techniques into deep discriminative models (e.g., DNNs) \
and deep probabilistic generative models (e.g., DBN, Deep Boltzmann Machine (DBM)). \
This classification scheme, however, misses a key insight gained in deep learning research about how generative \
models can greatly improve the training of DNNs and other deep discriminative models via better regularization.\
"""  # noqa: E501
    )

    for doc_id in doc_ids:
        docsearch._delete_doc(doc_id)


def test_vectara_mmr() -> None:
    """Test end to end construction and search."""

    # create a new Vectara instance
    docsearch: Vectara = Vectara()

    # start with some initial texts, added with add_texts
    texts = [
        """
        The way Grounded Generation with Vectara works is we only use valid responses 
        from your data relative to the search query. 
        This dramatically reduces hallucinations in Vectara's responses. 
        You can try it out on your own on our newly launched AskNews demo to 
        experience Grounded Generation, or register an account to ground generative 
        summaries on your own data.
        """,
        """
        Generative AI promises to revolutionize how you can benefit from your data, 
        but you need it to provide dependable information without the risk of data 
        leakage. This is why today we're adding a fundamental capability to our 
        platform to make generative AI safer to use. It enables you to ask your data 
        questions and get reliable, accurate answers by retrieving and summarizing 
        only the relevant information. We call it “Grounded Generation”. 
        """,
        """
        We are incredibly excited to share another feature with this launch: 
        Hybrid Search! Neural LLM systems are excellent at understanding the context 
        and meaning of end-user queries, but they can still underperform when matching 
        exact product SKUs, unusual names of people or companies, barcodes, and other 
        text which identifies entities rather than conveying semantics. We're bridging 
        this gap by introducing a lexical configuration that matches exact keywords, 
        supports Boolean operators, and executes phrase searches, and incorporates the 
        results into our neural search results.
        """,
    ]

    doc_ids = []
    for text in texts:
        ids = docsearch.add_documents([Document(page_content=text, metadata={})])
        doc_ids.extend(ids)

    # test max marginal relevance
    output1 = docsearch.max_marginal_relevance_search(
        "generative AI",
        k=2,
        fetch_k=6,
        lambda_mult=1.0,  # no diversity bias
        n_sentence_context=0,
    )
    assert len(output1) == 2
    assert (
        "Generative AI promises to revolutionize how you can benefit from your data"
        in output1[0].page_content
    )
    assert (
        "This is why today we're adding a fundamental capability to our platform"
        in output1[1].page_content
    )

    output2 = docsearch.max_marginal_relevance_search(
        "generative AI",
        k=2,
        fetch_k=6,
        lambda_mult=0.0,  # only diversity bias
        n_sentence_context=0,
    )
    assert len(output2) == 2
    assert (
        "Generative AI promises to revolutionize how you can benefit from your data"
        in output2[0].page_content
    )
    assert (
        ("Neural LLM systems are excellent at understanding the context and meaning "
         "of end-user queries")
        in output2[1].page_content
    )

    # delete the docs from the corpus
    for doc_id in doc_ids:
        docsearch._delete_doc(doc_id)
