import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import pytest
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import Callbacks
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import BaseLLM
from langchain.retrievers import WebResearchRetriever
from langchain.schema import BaseRetriever, Document
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.vectorstores import Chroma

from langchain_experimental.chains.qa_with_references.retrieval import (
    RetrievalQAWithReferencesChain,
)

logger = logging.getLogger(__name__)

samples = {
    # "apify": [
    #     "How do I use langchain?",
    # ]
    # "docs": [
    #     "How do I use langchain?",
    # ],
    # "google": [
    #     "what is the Machine learning?",
    #     "what is say about Matrix multiplication ?",
    #     "what is the most popular language among data scientists?",
    #     "How do I make money?",
    # ],
    "wikipedia": [
        "what is the Machine learning?",
        "what is say about Matrix multiplication ?",
        "what is the most popular language among data scientists?",
    ],
}

# %%
VERBOSE_PROMPT = False
VERBOSE_RESULT = False
USE_CACHE = True
CHUNK_SIZE = 500
CHUNK_OVERLAP = 5
TEMPERATURE = 0.0
MAX_TOKENS = 1500
REDUCE_K_BELOW_MAX_TOKENS = False
ALL_CHAIN_TYPE = ["stuff", "map_reduce", "refine", "map_rerank"]
ALL_SAMPLES = sorted({(k, v) for k, ls in samples.items() for v in ls})

# To test a selected combinaison, activate these values
# ALL_CHAIN_TYPE = [ "stuff",]
# ALL_SAMPLES = [("google", "how can i be better at football?")]

CALLBACKS: Callbacks = []

if VERBOSE_PROMPT or VERBOSE_RESULT:

    class ExStdOutCallbackHandler(StdOutCallbackHandler):
        def on_text(
            self,
            text: str,
            color: Optional[str] = None,
            end: str = "",
            **kwargs: Any,
        ) -> None:
            if VERBOSE_PROMPT:
                print("====")
                super().on_text(text=text, color=color, end=end)

        def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
            """Ajoute une trace des outputs du llm"""
            if VERBOSE_RESULT:
                print("\n\033[1m> Finished chain with\033[0m")
                knows_keys = {
                    "answer",
                    "output_text",
                    "text",
                    "result",
                    "outputs",
                    "output",
                }
                if "outputs" in outputs:
                    print("\n\033[33m")
                    print(
                        "\n---\n".join(
                            [text["text"].strip() for text in outputs["outputs"]]
                        )
                    )
                    print("\n\033[0m")
                elif knows_keys.intersection(outputs):
                    # Prend la première cles en intersection
                    print(
                        f"\n\033[33m{outputs[next(iter(knows_keys.intersection(outputs)))]}\n\033[0m"
                    )
                else:
                    pass

    CALLBACKS = [ExStdOutCallbackHandler()]


def _init_llm(temperature: float = TEMPERATURE, max_token: int = MAX_TOKENS) -> BaseLLM:
    import langchain
    from dotenv import load_dotenv
    from langchain.cache import SQLiteCache

    load_dotenv()

    if USE_CACHE:
        langchain.llm_cache = SQLiteCache(
            database_path="/tmp/cache_qa_with_reference.db"
        )
    llm = langchain.llms.OpenAI(
        temperature=temperature,
        max_tokens=max_token,
        # cache=False,
    )
    return llm


_cache_retriever: Dict[Tuple[str, str], BaseRetriever] = {}


def _get_retriever(
    provider: str, question: str, splitter: TextSplitter, llm: BaseLLM
) -> BaseRetriever:
    cache_retriever = _cache_retriever.get((provider, question))
    if cache_retriever:
        return cache_retriever
    import dotenv

    dotenv.load_dotenv(override=True)
    retriever: BaseRetriever
    import langchain

    loader: BaseLoader
    embeding_function: Embeddings = OpenAIEmbeddings()
    f = Path(tempfile.gettempdir(), "test_chroma.db")
    shutil.rmtree(f, ignore_errors=True)

    if provider == "wikipedia":
        retriever = langchain.retrievers.WikipediaRetriever(wiki_client=None)
        docs = retriever.get_relevant_documents(question)
        split_docs = splitter.split_documents(docs)
        vectorstore = Chroma(
            embedding_function=embeding_function,
            persist_directory=str(f.name),
        )
        vectorstore.add_documents(split_docs)
        retriever = vectorstore.as_retriever()

    elif provider == "google":
        if "GOOGLE_API_KEY" not in os.environ or "GOOGLE_CSE_ID" not in os.environ:
            pytest.skip("GOOGLE_API_KEY and GOOGLE_CSE_ID must be set")
        try:
            import googleapiclient  # type: ignore # noqa: F401
            import html2text  # type: ignore # noqa: F401

            # Search
            from langchain import GoogleSearchAPIWrapper

            # Initialize
            vectorstore = Chroma(
                embedding_function=embeding_function,
                persist_directory=str(f),
            )
            retriever = WebResearchRetriever.from_llm(
                vectorstore=vectorstore,
                llm=llm,
                search=GoogleSearchAPIWrapper(),
                text_splitter=cast(RecursiveCharacterTextSplitter, splitter),
            )
        except ImportError:
            pytest.skip("Use pip install google-api-python-client html2text")

    elif provider == "docs":
        loader = DirectoryLoader("../../", glob="**/*.md", loader_cls=TextLoader)
        index = VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            text_splitter=splitter,
            # embedding=_embedding,
            vectorstore_kwargs={},
        ).from_loaders([loader])
        retriever = index.vectorstore.as_retriever(search_kwargs={"k": 4})
    elif provider == "apify":
        if "APIFY_API_TOKEN" not in os.environ:
            pytest.skip("APIFY_API_TOKEN must be set")

        try:
            from langchain.utilities import ApifyWrapper

            apify = ApifyWrapper()
            loader = apify.call_actor(
                actor_id="apify/website-content-crawler",
                run_input={"startUrls": [{"url": "https://en.wikipedia.org/"}]},
                dataset_mapping_function=lambda item: Document(
                    page_content=item["text"] or "", metadata={"source": item["url"]}
                ),
            )
            index = VectorstoreIndexCreator(
                vectorstore_cls=Chroma,
                text_splitter=splitter,
                embedding=embeding_function,
                vectorstore_kwargs={},
            ).from_loaders([loader])
            retriever = index.vectorstore.as_retriever(search_kwargs={"k": 4})
        except ImportError:
            pytest.skip("Use pip install apify-client")
    else:
        raise ValueError()
    _cache_retriever[(provider, question)] = retriever
    return retriever


def _merge_result_by_urls(answer: Dict[str, Any]) -> Dict[str, List[str]]:
    references: Dict[str, List[str]] = {}
    for doc in answer["source_documents"]:
        source = doc.metadata.get("source", [])
        verbatims_for_source: List[str] = doc.metadata.get(source, [])
        verbatims_for_source.extend(doc.metadata.get("verbatims", []))
        references[source] = verbatims_for_source
    return references


def _test_qa_with_reference_chain(
    cls: Type,
    provider: str,
    chain_type: str,
    question: str,
    max_token: int,
    chunk_size: int,
    chunk_overlap: int,
    kwargs: Dict[str, Any] = {},
) -> None:
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        llm = _init_llm(max_token=max_token)
        retriever = _get_retriever(
            provider=provider,
            question=question,
            splitter=splitter,
            llm=llm,
        )

        qa_chain = cls.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            **kwargs,
        )
        answer = qa_chain(
            inputs={
                "question": question,
            },
            callbacks=CALLBACKS,
        )
        print(f'Question "{question}"\n' f'{answer["answer"]}\n\n')
        if "sources" in answer:
            # Old QA with sources
            print(f'Source "{answer["sources"]}"')
            for doc in answer.get("source_documents", []):
                print(f'- Doc {doc.metadata["source"]}')
        else:
            references = _merge_result_by_urls(answer)
            # Print the result
            for source, verbatims in references.items():
                print(f"Source {source}")
                for verbatim in verbatims:
                    print(f'-  "{verbatim}"')
    except ImportError as e:
        pytest.skip(f"Import error {e}")


@pytest.mark.parametrize("chain_type", ALL_CHAIN_TYPE)
# @pytest.mark.parametrize("provider,question",
#                          sorted({(k, l) for k, ls in samples.items() for l in ls}))
@pytest.mark.parametrize("provider,question", ALL_SAMPLES)
def test_qa_with_references_chain(
    provider: str, chain_type: str, question: str
) -> None:
    _test_qa_with_reference_chain(
        cls=RetrievalQAWithReferencesChain,
        provider=provider,
        chain_type=chain_type,
        question=question,
        max_token=2000,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        kwargs={
            "reduce_k_below_max_tokens": REDUCE_K_BELOW_MAX_TOKENS,
        },
    )
