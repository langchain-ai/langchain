"""Filter that uses an LLM to drop documents that aren't relevant to the query."""
from typing import Any, Callable, Dict, List, Optional

from langchain import BasePromptTemplate, LLMChain, PromptTemplate
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.retrievers.document_filters.base import (
    BaseDocumentFilter,
    RetrievedDocument,
)
from langchain.retrievers.document_filters.relevant_chain_prompt import prompt_template
from langchain.schema import BaseLanguageModel, Document


def _get_default_chain_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"],
        output_parser=BooleanOutputParser(),
    )


def default_get_input(query: str, doc: Document) -> Dict[str, Any]:
    """Return the compression chain input."""
    return {"question": query, "context": doc.page_content}


class LLMChainDocumentFilter(BaseDocumentFilter):
    """Filter that drops documents that aren't relevant to the query."""

    llm_chain: LLMChain
    """LLM wrapper to use for filtering documents. 
    The chain prompt is expected to have a BooleanOutputParser."""

    get_input: Callable[[str, Document], dict] = default_get_input
    """Callable for constructing the chain input from the query and a Document."""

    def filter(
        self, docs: List[RetrievedDocument], query: str
    ) -> List[RetrievedDocument]:
        """Filter down documents based on their relevance to the query."""
        filtered_docs = []
        for doc in docs:
            _input = self.get_input(query, doc)
            include_doc = self.llm_chain.predict_and_parse(**_input)
            if include_doc:
                filtered_docs.append(doc)
        return filtered_docs

    async def afilter(
        self, docs: List[RetrievedDocument], query: str
    ) -> List[RetrievedDocument]:
        """Filter down documents."""
        raise NotImplementedError

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any
    ) -> "LLMChainDocumentFilter":
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        llm_chain = LLMChain(llm=llm, prompt=_prompt)
        return cls(llm_chain=llm_chain, **kwargs)
