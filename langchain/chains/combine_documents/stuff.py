"""Chain that combines documents by stuffing into context."""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate


def _get_default_document_prompt() -> PromptTemplate:
    return PromptTemplate(input_variables=["page_content"], template="{page_content}")


class StuffDocumentsChain(BaseCombineDocumentsChain, BaseModel):
    """Chain that combines documents by stuffing into context."""

    llm_chain: LLMChain
    """LLM wrapper to use after formatting documents."""
    document_prompt: BasePromptTemplate = Field(
        default_factory=_get_default_document_prompt
    )
    """Prompt to use to format each document."""
    document_variable_name: str
    """The variable name in the llm_chain to put the documents in.
    If only one variable in the llm_chain, this need not be provided."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def get_default_document_variable_name(cls, values: Dict) -> Dict:
        """Get default document variable name, if not provided."""
        if "document_variable_name" not in values:
            llm_chain_variables = values["llm_chain"].prompt.input_variables
            if len(llm_chain_variables) == 1:
                values["document_variable_name"] = llm_chain_variables[0]
            else:
                raise ValueError(
                    "document_variable_name must be provided if there are "
                    "multiple llm_chain_variables"
                )
        else:
            llm_chain_variables = values["llm_chain"].prompt.input_variables
            if values["document_variable_name"] not in llm_chain_variables:
                raise ValueError(
                    f"document_variable_name {values['document_variable_name']} was "
                    f"not found in llm_chain input_variables: {llm_chain_variables}"
                )
        return values

    def _get_inputs(self, docs: List[Document], **kwargs: Any) -> dict:
        # Get relevant information from each document.
        doc_dicts = []
        for doc in docs:
            base_info = {"page_content": doc.page_content}
            base_info.update(doc.metadata)
            document_info = {
                k: base_info[k] for k in self.document_prompt.input_variables
            }
            doc_dicts.append(document_info)
        # Format each document according to the prompt
        doc_strings = [self.document_prompt.format(**doc) for doc in doc_dicts]
        # Join the documents together to put them in the prompt.
        inputs = kwargs.copy()
        inputs[self.document_variable_name] = "\n\n".join(doc_strings)
        return inputs

    def prompt_length(self, docs: List[Document], **kwargs: Any) -> Optional[int]:
        """Get the prompt length by formatting the prompt."""
        inputs = self._get_inputs(docs, **kwargs)
        prompt = self.llm_chain.prompt.format(**inputs)
        return self.llm_chain.llm.get_num_tokens(prompt)

    def combine_docs(self, docs: List[Document], **kwargs: Any) -> Tuple[str, dict]:
        """Stuff all documents into one prompt and pass to LLM."""
        inputs = self._get_inputs(docs, **kwargs)
        # Call predict on the LLM.
        return self.llm_chain.predict(**inputs), {}
