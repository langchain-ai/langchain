from __future__ import annotations
from typing import Optional, Any, Union, Literal, List

from plistlib import Dict

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.research.readers import DocReadingChain, ParallelApplyChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.research.search import GenericSearcher
from langchain.text_splitter import TextSplitter
from langchain.chains.research.fetch import PlaywrightDownloadHandler, DownloadHandler


class Research(Chain):
    """A simple research chain."""
    searcher: GenericSearcher
    """The searcher to use to search for documents."""
    reader: Chain
    """The reader to use to read documents and produce an answer."""
    downloader: DownloadHandler

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return ["docs", "summary"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run the chain asynchronously."""
        question = inputs["question"]
        search_results = self.searcher({"question": question})
        urls = search_results["urls"]
        blobs = self.downloader.download(urls)
        raise NotImplementedError()

    @classmethod
    def from_llms(
            cls,
            link_selection_llm: BaseLanguageModel,
            query_generation_llm: BaseLanguageModel,
            qa_chain: LLMChain,
            *,
            top_k_per_search: int = -1,
            max_concurrency: int = 1,
            max_num_pages_per_doc: int = 100,
            text_splitter: Union[TextSplitter, Literal["recursive"]] = "recursive",
    ) -> Research:
        """Helper to create a research chain from standard llm related components.

        Args:
            link_selection_llm: The language model to use for link selection.
            query_generation_llm: The language model to use for query generation.
            qa_chain: The chain to use to answer the question.
            top_k_per_search: The number of documents to return per search.
            max_concurrency: The maximum number of concurrent reads.
            max_num_pages_per_doc: The maximum number of pages to read per document.
            text_splitter: The text splitter to use to split the document into smaller chunks.

        Returns:
            A research chain.
        """
        searcher = GenericSearcher.from_llms(
            link_selection_llm,
            query_generation_llm,
            top_k_per_search=top_k_per_search,
        )
        if isinstance(text_splitter, str):
            if text_splitter == "recursive":
                _text_splitter = RecursiveCharacterTextSplitter()
            else:
                raise ValueError(f"Invalid text splitter: {text_splitter}")
        elif isinstance(text_splitter, TextSplitter):
            _text_splitter = text_splitter
        else:
            raise TypeError(f"Invalid text splitter: {type(text_splitter)}")
        reader = ParallelApplyChain(
            chain=DocReadingChain(
                qa_chain, max_num_pages_per_doc=max_num_pages_per_doc,
                text_splitter=text_splitter
            ),
            max_concurrency=max_concurrency,
        )
        return cls(searcher=searcher, reader=reader)
