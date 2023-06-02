"""Module contains supporting chains for research use case."""
import asyncio
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.schema import Document
from langchain.text_splitter import TextSplitter


class DocReadingChain(Chain):
    """A reader chain should use one of the QA chains to answer a question.

    This chain is also responsible for splitting the document into smaller chunks
    and then passing the chunks to an underlying QA chain.

    A brute force chain that reads an entire document (or the first N pages).
    """

    chain: Chain
    """The chain to use to answer the question."""

    text_splitter: TextSplitter
    """The text splitter to use to split the document into smaller chunks."""

    max_num_docs: int
    """The maximum number of documents to split the document into.
    
    Use -1 to denote no limit to the number of pages to read.
    """

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return ["doc", "question"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return ["answer"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Process a long document synchronously."""
        source_document = inputs["doc"]
        question = inputs["question"]

        sub_docs = self.text_splitter.split_documents([source_document])
        if self.max_num_docs > 0:
            _sub_docs = sub_docs[: self.max_num_docs]
        else:
            _sub_docs = sub_docs

        response = self.chain(
            {"input_documents": _sub_docs, "question": question},
            callbacks=run_manager.get_child() if run_manager else None,
        )
        summary_doc = Document(
            page_content=response["output_text"],
            metadata=source_document.metadata,
        )
        return {"answer": summary_doc}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Process a long document asynchronously."""
        source_document = inputs["doc"]
        question = inputs["question"]
        sub_docs = self.text_splitter.split_documents([source_document])
        if self.max_num_docs > 0:
            _sub_docs = sub_docs[: self.max_num_docs]
        else:
            _sub_docs = sub_docs

        results = await self.chain.acall(
            {"input_documents": _sub_docs, "question": question},
            callbacks=run_manager.get_child() if run_manager else None,
        )
        summary_doc = Document(
            page_content=results["output_text"],
            metadata=source_document.metadata,
        )

        return {"answer": summary_doc}


class ParallelApplyChain(Chain):
    """Utility chain to apply a given chain in parallel across input documents.

    This chain needs to handle a limit on concurrency.

    WARNING: Parallelization only implemented on the async path.
    """

    chain: Chain

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return ["inputs"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys."""
        return ["results"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run the chain."""
        # TODO(): parallelize this
        chain_inputs = inputs["inputs"]

        results = [
            self.chain(
                chain_input,
                callbacks=run_manager.get_child() if run_manager else None,
            )
            for chain_input in chain_inputs
        ]
        return {"results": results}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run the chain."""
        chain_inputs = inputs["inputs"]

        results = await asyncio.gather(
            *[
                self.chain.acall(
                    chain_input,
                    callbacks=run_manager.get_child() if run_manager else None,
                )
                for chain_input in chain_inputs
            ]
        )
        return {"results": results}
