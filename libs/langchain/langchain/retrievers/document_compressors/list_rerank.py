"""Filter that uses an LLM to rerank documents listwise and select top-k."""
import logging
from typing import Any, Callable, Dict, Optional, Sequence

from langchain.callbacks.manager import Callbacks
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.schema import BasePromptTemplate, Document
from langchain.schema.language_model import BaseLanguageModel

logger = logging.getLogger(__name__)


def _get_default_chain_prompt() -> PromptTemplate:
    prompt_template = """
{context}
Query = ```{question}```
Sort the Documents by their relevance to the Query.

{format_instructions}
Sorted Documents:
    """
    description = (
        """Reranked documents. Format: {"document_id": <int>, "score": <number>}"""
    )
    response_schemas = [
        ResponseSchema(
            name="reranked_documents",
            description=description,
            type="array[dict]",
        )
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    return PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"],
        output_parser=output_parser,
        partial_variables={"format_instructions": format_instructions},
    )


def default_get_input(query: str, documents: Sequence[Document]) -> Dict[str, Any]:
    """Return the compression chain input."""
    context = ""
    for index, doc in enumerate(documents):
        context += f"Document ID: {index} ```{doc.page_content}```\n"
    context += f"Documents = [Document ID: 0, ..., Document ID: {len(documents) - 1}]"
    return {"question": query, "context": context}


class ListRerank(BaseDocumentCompressor):
    """
    Document compressor that uses `Zero-Shot Listwise Document Reranking`.

    Source: https://arxiv.org/pdf/2305.02156.pdf
    """

    top_n: int = 3
    """Number of documents to return."""

    llm_chain: LLMChain
    """LLM wrapper to use for filtering documents."""

    get_input: Callable[[str, Document], dict] = default_get_input
    """Callable for constructing the chain input from the query and Documents."""

    fallback: bool = False
    """Whether to fallback to the original document scores if the LLM fails."""

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Filter down documents based on their relevance to the query."""
        _input = self.get_input(query, documents)
        try:
            results = self.llm_chain.predict_and_parse(**_input, callbacks=callbacks)
            top_documents = results["reranked_documents"][: self.top_n]
        except Exception as e:
            return self._handle_exception(documents, e)

        final_results = []
        for r in top_documents:
            try:
                doc = documents[r["document_id"]]
                score = float(r["score"])
            except Exception as e:
                return self._handle_exception(documents, e)

            doc.metadata["relevance_score"] = score
            final_results.append(doc)
        return final_results

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> "ListRerank":
        """Create a ListRerank document compressor from a language model.

        Args:
            llm: The language model to use for filtering.
            prompt: The prompt to use for the filter.
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            A ListRerank document compressor that uses the given language model.
        """
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        llm_chain = LLMChain(llm=llm, prompt=_prompt)
        return cls(llm_chain=llm_chain, **kwargs)

    def _handle_exception(
        self, documents: Sequence[Document], exception: Exception
    ) -> Sequence[Document]:
        """Return the top documents by original ranking or raise an exception."""
        if self.fallback:
            logger.warning(
                "Failed to generate or parse LLM response. "
                "Falling back to original scores."
            )
            return documents[: self.top_n]
        else:
            raise exception
