"""Retriever tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.prompts import (
    BasePromptTemplate,
    PromptTemplate,
    aformat_document,
    format_document,
)
from langchain_core.tools.structured import StructuredTool

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever


class RetrieverInput(BaseModel):
    """Input to the retriever."""

    query: str = Field(description="query to look up in retriever")


def create_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    *,
    document_prompt: BasePromptTemplate | None = None,
    document_separator: str = "\n\n",
    response_format: Literal["content", "content_and_artifact"] = "content",
) -> StructuredTool:
    r"""Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.
        document_prompt: The prompt to use for the document.
        document_separator: The separator to use between documents.
        response_format: The tool response format.

            If `"content"` then the output of the tool is interpreted as the contents of
            a `ToolMessage`. If `"content_and_artifact"` then the output is expected to
            be a two-tuple corresponding to the `(content, artifact)` of a `ToolMessage`
            (artifact being a list of documents in this case).

    Returns:
        Tool class to pass to an agent.
    """
    document_prompt_ = document_prompt or PromptTemplate.from_template("{page_content}")

    def func(
        query: str, callbacks: Callbacks = None
    ) -> str | tuple[str, list[Document]]:
        docs = retriever.invoke(query, config={"callbacks": callbacks})
        content = document_separator.join(
            format_document(doc, document_prompt_) for doc in docs
        )
        if response_format == "content_and_artifact":
            return (content, docs)
        return content

    async def afunc(
        query: str, callbacks: Callbacks = None
    ) -> str | tuple[str, list[Document]]:
        docs = await retriever.ainvoke(query, config={"callbacks": callbacks})
        content = document_separator.join(
            [await aformat_document(doc, document_prompt_) for doc in docs]
        )
        if response_format == "content_and_artifact":
            return (content, docs)
        return content

    return StructuredTool(
        name=name,
        description=description,
        func=func,
        coroutine=afunc,
        args_schema=RetrieverInput,
        response_format=response_format,
    )
