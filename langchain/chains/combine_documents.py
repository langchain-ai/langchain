"""Document combining chain."""

from typing import Any, Dict, List

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import Prompt


def _get_default_document_prompt() -> Prompt:
    return Prompt(input_variables=["page_content"], template="{page_content}")


class CombineDocumentsChain(Chain, BaseModel):
    """Combine documents."""

    llm_chain: LLMChain
    """LLM wrapper to use after formatting documents."""
    document_prompt: BasePromptTemplate = Field(
        default_factory=_get_default_document_prompt
    )
    """Prompt to use to format each document."""
    document_variable_name: str
    """The variable name in the llm_chain to put the documents in.
    If only one variable in the llm_chain, this need not be provided."""
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

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        docs = inputs[self.input_key]
        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
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
        other_keys[self.document_variable_name] = "\n".join(doc_strings)
        # Call predict on the LLM.
        output = self.llm_chain.predict(**other_keys)
        return {self.output_key: output}
