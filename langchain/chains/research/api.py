from __future__ import annotations

import itertools
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.research.download import AutoDownloadHandler, DownloadHandler
from langchain.chains.research.readers import DocReadingChain, ParallelApplyChain
from langchain.chains.research.search import GenericSearcher
from langchain.document_loaders.parsers.html.markdownify import MarkdownifyHTMLParser
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


class Research(Chain):
    """A research chain.

    A research chain is composed of the following components:

    1. A searcher that searches for documents using a search engine.
        - The searcher is responsible to return a list of URLs of documents that
          may be relevant to read to be able to answer the question.
    2. A downloader that downloads the documents.
    3. An HTML to markdown parser (hard coded) that converts the HTML to markdown.
        * Conversion to markdown is lossy
        * However, it can significantly reduce the token count of the document
        * Markdown helps to preserve some styling information
          (e.g., bold, italics, links, headers) which is expected to help the reader
          to answer certain kinds of questions correctly.
    4. A reader that reads the documents and produces an answer.

    Limitations:
        * This research chain only implements a single hop at the moment; i.e.,
          it goes from the questions to a list of URLs to documents to compiling
          answers.
        * The reader chain needs to match the task. For example, if using a QA refine
          chain, a task of collecting a list of entries from a long document will
          fail because the QA refine chain is not designed to handle such a task.

    The chain can be extended to continue crawling the documents in attempt
    to discover relevant pages that were not surfaced by the search engine.

    Amongst other problems without continuing the crawl, it is impossible to
    continue getting results from pages that involve pagination.
    """

    searcher: GenericSearcher
    """The searcher to use to search for documents."""
    reader: Chain
    """The reader to use to read documents and produce an answer."""
    downloader: DownloadHandler
    """The downloader to use to download the documents.
    
    A few different implementations of the download handler have been provided.
    
    Keep in mind that some websites require execution of JavaScript to load
    the DOM.
    """

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return ["docs"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run the chain synchronously."""
        question = inputs["question"]
        search_results = self.searcher(
            {"question": question},
            callbacks=run_manager.get_child() if run_manager else None,
        )
        urls = search_results["urls"]
        blobs = self.downloader.download(urls)
        parser = MarkdownifyHTMLParser()
        docs = itertools.chain.from_iterable(
            parser.lazy_parse(blob) for blob in blobs if blob is not None
        )
        _inputs = [{"doc": doc, "question": question} for doc in docs]
        results = self.reader(
            _inputs, callbacks=run_manager.get_child() if run_manager else None
        )
        return {
            "docs": [result["answer"] for result in results["inputs"]],
        }

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run the chain asynchronously."""
        question = inputs["question"]
        search_results = await self.searcher.acall(
            {"question": question},
            callbacks=run_manager.get_child() if run_manager else None,
        )
        urls = search_results["urls"]
        blobs = await self.downloader.adownload(urls)
        parser = MarkdownifyHTMLParser()
        docs = itertools.chain.from_iterable(
            parser.lazy_parse(blob) for blob in blobs if blob is not None
        )
        _inputs = [{"doc": doc, "question": question} for doc in docs]
        results = await self.reader.acall(
            _inputs,
            callbacks=run_manager.get_child() if run_manager else None,
        )
        return {
            "docs": [result["answer"] for result in results["results"]],
        }

    @classmethod
    def from_llms(
        cls,
        *,
        query_generation_llm: BaseLanguageModel,
        link_selection_llm: BaseLanguageModel,
        underlying_reader_chain: LLMChain,
        top_k_per_search: int = -1,
        max_concurrency: int = 1,
        max_num_pages_per_doc: int = 5,
        text_splitter: Union[TextSplitter, Literal["recursive"]] = "recursive",
        download_handler: Union[DownloadHandler, Literal["auto"]] = "auto",
        text_splitter_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Research:
        """Helper to create a research chain from standard llm related components.

        Args:
            query_generation_llm: The language model to use for query generation.
            link_selection_llm: The language model to use for link selection.
            underlying_reader_chain: The chain to use to answer the question.
            top_k_per_search: The number of documents to return per search.
            max_concurrency: The maximum number of concurrent reads.
            max_num_pages_per_doc: The maximum number of pages to read per document.
            text_splitter: The text splitter to use to split the document into
                           smaller chunks.
            download_handler: The download handler to use to download the documents.
                              Provide either a download handler or the name of a
                              download handler.
                              - "auto" swaps between using requests and playwright
            text_splitter_kwargs: The keyword arguments to pass to the text splitter.
                                  Only use when providing a text splitter as string.

        Returns:
            A research chain.
        """
        if isinstance(text_splitter, str):
            if text_splitter == "recursive":
                _text_splitter_kwargs = text_splitter_kwargs or {}
                _text_splitter: TextSplitter = RecursiveCharacterTextSplitter(
                    **_text_splitter_kwargs
                )
            else:
                raise ValueError(f"Invalid text splitter: {text_splitter}")
        elif isinstance(text_splitter, TextSplitter):
            _text_splitter = text_splitter
        else:
            raise TypeError(f"Invalid text splitter: {type(text_splitter)}")

        if isinstance(download_handler, str):
            if download_handler == "auto":
                _download_handler: DownloadHandler = AutoDownloadHandler()
            else:
                raise ValueError(f"Invalid download handler: {download_handler}")
        elif isinstance(download_handler, DownloadHandler):
            _download_handler = download_handler
        else:
            raise TypeError(f"Invalid download handler: {type(download_handler)}")

        searcher = GenericSearcher.from_llms(
            link_selection_llm,
            query_generation_llm,
            top_k_per_search=top_k_per_search,
        )

        doc_reading_chain = DocReadingChain(
            chain=underlying_reader_chain,
            max_num_docs=max_num_pages_per_doc,
            text_splitter=_text_splitter,
        )
        # Can read multiple documents in parallel
        multi_reader = ParallelApplyChain(
            chain=doc_reading_chain,
            max_concurrency=max_concurrency,
        )
        return cls(searcher=searcher, reader=multi_reader, downloader=_download_handler)
