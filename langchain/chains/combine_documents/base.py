"""Base interface for chains combining documents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain.chains.base import Chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


class BaseCombineDocumentsChain(Chain, BaseModel, ABC):
    """Base interface for chains combining documents."""

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

        Returns None if the method does not depend on the prompt length.
        """
        return None

    @abstractmethod
    def combine_docs(self, docs: List[Document], **kwargs: Any) -> Tuple[str, dict]:
        """Combine documents into a single string."""

    @abstractmethod
    async def acombine_docs(
        self, docs: List[Document], **kwargs: Any
    ) -> Tuple[str, dict]:
        """Combine documents into a single string asynchronously."""

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        docs = inputs[self.input_key]
        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        output, extra_return_dict = self.combine_docs(docs, **other_keys)
        extra_return_dict[self.output_key] = output
        return extra_return_dict

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        docs = inputs[self.input_key]
        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        output, extra_return_dict = await self.acombine_docs(docs, **other_keys)
        extra_return_dict[self.output_key] = output
        return extra_return_dict


class AnalyzeDocumentChain(Chain, BaseModel):
    """Chain that splits documents, then analyzes it in pieces."""

    input_key: str = "input_document"  #: :meta private:
    output_key: str = "output_text"  #: :meta private:
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
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        document = inputs[self.input_key]
        docs = self.text_splitter.create_documents([document])
        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        other_keys[self.combine_docs_chain.input_key] = docs
        return self.combine_docs_chain(other_keys, return_only_outputs=True)
