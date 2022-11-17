"""Document combining chain."""

from typing import Dict, List, Any

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.llms.base import LLM
from langchain.prompts.base import BasePrompt
from langchain.text_splitter import TextSplitter


class CombineDocumentsChain(Chain, BaseModel):
    """Combine documents."""

    llm_chain: LLMChain
    """LLM wrapper to use after formatting documents."""
    document_prompt: BasePrompt
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
        doc_dicts = [{k: doc.dict()[k] for k in self.document_prompt.input_variables} for doc in inputs[self.input_key]]
        doc_strings = [self.document_prompt.format(**doc) for doc in doc_dicts]
        doc_variable = self.llm_chain.prompt.input_variables[0]
        output = self.llm_chain.predict(**{doc_variable: "\n".join(doc_strings)})
        return {self.output_key: output}
