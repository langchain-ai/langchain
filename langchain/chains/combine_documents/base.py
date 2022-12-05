"""Document combining chain."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel

from langchain.chains.base import Chain
from langchain.docstore.document import Document
from langchain.prompts.prompt import Prompt


def _get_default_document_prompt() -> Prompt:
    return Prompt(input_variables=["page_content"], template="{page_content}")


class BaseCombineDocumentsChain(Chain, BaseModel, ABC):
    """Combine documents."""

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

    @abstractmethod
    def combine_docs(self, docs: List[Document], **kwargs: Any) -> str:
        """Combine documents."""

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        docs = inputs[self.input_key]
        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        output = self.combine_docs(docs, **other_keys)
        return {self.output_key: output}
