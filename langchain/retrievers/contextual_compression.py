""""""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, Field, root_validator

from langchain import BasePromptTemplate, LLMChain, PromptTemplate
from langchain.schema import BaseLanguageModel, BaseRetriever, Document

prompt_template = """Write a concise summary of the following text. Make note of any key entities and words:

"{context}"

CONCISE SUMMARY:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context"])


def _get_default_document_prompt() -> PromptTemplate:
    return PromptTemplate(input_variables=["page_content"], template="{page_content}")


class ContextualCompressionRetriever(BaseRetriever, BaseModel):
    """"""

    llm_chain: LLMChain
    """LLM wrapper to use for compressing documents."""
    base_retriever: BaseRetriever
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

    def _get_input(self, doc: Document) -> Dict[str, Any]:
        """"""
        all_info = {"page_content": doc.page_content, **doc.metadata}
        doc_info = {k: all_info[k] for k in self.document_prompt.input_variables}
        doc_str = self.document_prompt.format(**doc_info)
        return {self.document_variable_name: doc_str}

    def _compress_docs(self, docs: List[Document]) -> List[Document]:
        """Compress page content of raw documents."""
        compressed_docs = []
        for doc in docs:
            input = self._get_input(doc)
            output = self.llm_chain.predict(**input)
            compressed_docs.append(Document(page_content=output, metadata=doc.metadata))
        return compressed_docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        docs = self.base_retriever.get_relevant_documents(query)
        compressed_docs = self._compress_docs(docs)
        return compressed_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        raise NotImplementedError(
            "Async `aget_relevant_documents` function not implemented yet."
        )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        base_retriever: BaseRetriever,
        prompt: Optional[PromptTemplate] = None,
        **kwargs: Any,
    ) -> "ContextualCompressionRetriever":
        """Initialize from LLM."""
        _prompt = prompt if prompt is not None else PROMPT
        llm_chain = LLMChain(llm=llm, prompt=_prompt)
        return cls(llm_chain=llm_chain, base_retriever=base_retriever, **kwargs)
