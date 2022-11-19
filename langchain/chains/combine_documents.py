"""Document combining chain."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, Field

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.llms.base import LLM
from langchain.prompts.base import BasePrompt
from langchain.prompts.prompt import Prompt
from langchain.text_splitter import TextSplitter


def _get_default_document_prompt():
    return Prompt(input_variables=["page_content"], template="{page_content}")


class CombineDocumentsChain(Chain, BaseModel):
    """Combine documents."""

    llm_chain: LLMChain
    """LLM wrapper to use after formatting documents."""
    document_prompt: BasePrompt = Field(default_factory=_get_default_document_prompt)
    """Prompt to use to format each document."""
    input_key: str = "input_documents"  #: :meta private:
    output_key: str = "output_text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

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
        docs = inputs[self.input_key]
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        doc_dicts = []
        for doc in docs:
            base_info = {"page_content": doc.page_content}
            base_info.update(doc.metadata)
            doc_dicts.append(
                {k: base_info[k] for k in self.document_prompt.input_variables}
            )
        doc_strings = [self.document_prompt.format(**doc) for doc in doc_dicts]
        doc_variable = self.llm_chain.prompt.input_variables[0]
        other_keys[doc_variable] = "\n\n".join(doc_strings)
        output = self.llm_chain.predict(**other_keys)
        return {self.output_key: output}
