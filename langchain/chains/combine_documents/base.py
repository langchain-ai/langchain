"""Base interface for chains combining documents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


class BaseCombineDocumentsChain(Chain, ABC):
    """Base interface for chains combining documents.

    Subclasses of this chain deal with combining documents in a variety of
    ways. This base class exists to add some uniformity in the interface these types
    of chains should expose. Namely, they expect an input key related to the documents
    to use (default `input_documents`), and then also expose a method to calculate
    the length of a prompt from documents (useful for outside callers to use to
    determine whether it's safe to pass a list of documents into this chain or whether
    that will longer than the context length).
    """

    input_key: str = "input_documents"  #: :meta private:
    output_key: str = "output_text"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    def prompt_length(self, docs: List[Document], **kwargs: Any) -> Optional[int]:
        """Return the prompt length given the documents passed in.

        This can be used by a caller to determine whether passing in a list
        of documents would exceed a certain prompt length. This useful when
        trying to ensure that the size of a prompt remains below a certain
        context limit.

        Args:
            docs: List[Document], a list of documents to use to calculate the
                total prompt length.

        Returns:
            Returns None if the method does not depend on the prompt length,
            otherwise the length of the prompt in tokens.
        """
        return None

    @abstractmethod
    def combine_docs(self, docs: List[Document], **kwargs: Any) -> Tuple[str, dict]:
        """Combine documents into a single string.

        Args:
            docs: List[Document], the documents to combine
            **kwargs: Other parameters to use in combining documents, often
                other inputs to the prompt.

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """

    @abstractmethod
    async def acombine_docs(
        self, docs: List[Document], **kwargs: Any
    ) -> Tuple[str, dict]:
        """Combine documents into a single string.

        Args:
            docs: List[Document], the documents to combine
            **kwargs: Other parameters to use in combining documents, often
                other inputs to the prompt.

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """

    def _call(
        self,
        inputs: Dict[str, List[Document]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Prepare inputs, call combine docs, prepare outputs."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        docs = inputs[self.input_key]
        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        output, extra_return_dict = self.combine_docs(
            docs, callbacks=_run_manager.get_child(), **other_keys
        )
        extra_return_dict[self.output_key] = output
        return extra_return_dict

    async def _acall(
        self,
        inputs: Dict[str, List[Document]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Prepare inputs, call combine docs, prepare outputs."""
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        docs = inputs[self.input_key]
        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        output, extra_return_dict = await self.acombine_docs(
            docs, callbacks=_run_manager.get_child(), **other_keys
        )
        extra_return_dict[self.output_key] = output
        return extra_return_dict


class AnalyzeDocumentChain(Chain):
    """Chain that splits documents, then analyzes it in pieces.

    This chain is parameterized by a TextSplitter and a CombineDocumentsChain.
    This chain takes a single document as input, and then splits it up into chunks
    and then passes those chucks to the CombineDocumentsChain.
    """

    input_key: str = "input_document"  #: :meta private:
    text_splitter: TextSplitter = Field(default_factory=RecursiveCharacterTextSplitter)
    combine_docs_chain: BaseCombineDocumentsChain

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return self.combine_docs_chain.output_keys

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Split document into chunks and pass to CombineDocumentsChain."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        document = inputs[self.input_key]
        docs = self.text_splitter.create_documents([document])
        # Other keys are assumed to be needed for LLM prediction
        other_keys: Dict = {k: v for k, v in inputs.items() if k != self.input_key}
        other_keys[self.combine_docs_chain.input_key] = docs
        return self.combine_docs_chain(
            other_keys, return_only_outputs=True, callbacks=_run_manager.get_child()
        )
