"""Base interface for chains combining documents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from langchain.chains.base import Chain
from langchain.docstore.document import Document


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
    def combine_docs(self, docs: List[Document], **kwargs: Any) -> str:
        """Combine documents into a single string."""

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        docs = inputs[self.input_key]
        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        output = self.combine_docs(docs, **other_keys)
        return {self.output_key: output}
