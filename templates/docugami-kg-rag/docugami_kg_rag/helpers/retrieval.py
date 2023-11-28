import re
from typing import List, Optional

import pinecone
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain.tools.base import BaseTool
from langchain.vectorstores.pinecone import Pinecone

from docugami_kg_rag.config import (
    EMBEDDINGS,
    LLM,
    PINECONE_INDEX,
    RETRIEVER_K,
    SMALL_FRAGMENT_MAX_TEXT_LENGTH,
    LocalIndexState,
)
from docugami_kg_rag.helpers.fused_summary_retriever import (
    FusedSummaryRetriever,
    SearchType,
)
from docugami_kg_rag.helpers.prompts import (
    ASSISTANT_SYSTEM_MESSAGE,
    CREATE_DIRECT_RETRIEVAL_TOOL_DESCRIPTION_PROMPT,
)


def docset_name_to_direct_retriever_tool_function_name(name: str) -> str:
    """
    Converts a docset name to a direct retriever tool function name.

    Direct retriever tool function names follow these conventions:
    1. Retrieval tool function names always start with "search_".
    2. The rest of the name should be a lowercased string, with underscores for whitespace.
    3. Exclude any characters other than a-z (lowercase) from the function name, replacing them with underscores.
    4. The final function name should not have more than one underscore together.

    >>> docset_name_to_direct_retriever_tool_function_name('Earnings Calls')
    'search_earnings_calls'
    >>> docset_name_to_direct_retriever_tool_function_name('COVID-19   Statistics')
    'search_covid_19_statistics'
    >>> docset_name_to_direct_retriever_tool_function_name('2023 Market Report!!!')
    'search_2023_market_report'
    """
    # Replace non-letter characters with underscores and remove extra whitespaces
    name = re.sub(r"[^a-z\d]", "_", name.lower())
    # Replace whitespace with underscores and remove consecutive underscores
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    name = name.strip("_")

    return f"search_{name}"


def chunks_to_direct_retriever_tool_description(name: str, chunks: List[Document]):
    """
    Converts a set of chunks to a direct retriever tool description.
    """
    texts = [c.page_content for c in chunks[:100]]
    doc_fragment = "\n".join(texts)[:SMALL_FRAGMENT_MAX_TEXT_LENGTH]

    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", ASSISTANT_SYSTEM_MESSAGE),
                ("human", CREATE_DIRECT_RETRIEVAL_TOOL_DESCRIPTION_PROMPT),
            ]
        )
        | LLM
        | StrOutputParser()
    )
    summary = chain.invoke({"docset_name": name, "doc_fragment": doc_fragment})
    return f"Searches for and returns chunks from {name} documents. {summary}"


def get_retrieval_tool_for_docset(
    docset_id: str, docset_state: LocalIndexState
) -> Optional[BaseTool]:
    # Chunks are in the vector store, and full documents are in the store inside the local state

    docset_pinecone_index_name = f"{PINECONE_INDEX}-{docset_id}"
    if docset_pinecone_index_name not in pinecone.list_indexes():
        return None

    chunk_vectorstore = Pinecone.from_existing_index(
        docset_pinecone_index_name, EMBEDDINGS
    )
    retriever = FusedSummaryRetriever(
        vectorstore=chunk_vectorstore,
        summarystore=docset_state.doc_summaries_by_id,
        search_kwargs={"k": RETRIEVER_K},
        search_type=SearchType.mmr,
    )

    return create_retriever_tool(
        retriever=retriever,
        name=docset_state.retrieval_tool_function_name,
        description=docset_state.retrieval_tool_description,
    )
