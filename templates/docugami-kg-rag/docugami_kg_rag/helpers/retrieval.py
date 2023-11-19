import re
from typing import Dict, List, Optional

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
    CREATE_TOOL_DESCRIPTION_PROMPT,
)


def docset_name_to_retriever_tool_function_name(name: str) -> str:
    """
    Converts a docset name to a retriever tool function name.

    Retriever tool function names follow these conventions:
    1. Retrieval tool function names always start with "search_".
    2. The rest of the name should be a lowercased string, with underscores for whitespace.
    3. Exclude any characters other than a-z (lowercase) from the function name, replacing them with underscores.
    4. The final function name should not have more than one underscore together.

    >>> docset_name_to_retriever_tool_function_name('Earnings Calls')
    'search_earnings_calls'
    >>> docset_name_to_retriever_tool_function_name('COVID-19   Statistics')
    'search_covid_19_statistics'
    >>> docset_name_to_retriever_tool_function_name('2023 Market Report!!!')
    'search_2023_market_report'
    """
    # Replace non-letter characters with underscores and remove extra whitespaces
    name = re.sub(r"[^a-z\d]", "_", name.lower())
    # Replace whitespace with underscores and remove consecutive underscores
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    name = name.strip("_")

    return f"search_{name}"


def chunks_to_retriever_tool_description(name: str, chunks: List[Document]):
    texts = [c.page_content for c in chunks[:100]]
    doc_fragment = "\n".join(texts)[:SMALL_FRAGMENT_MAX_TEXT_LENGTH]

    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", ASSISTANT_SYSTEM_MESSAGE),
                ("human", CREATE_TOOL_DESCRIPTION_PROMPT),
            ]
        )
        | LLM
        | StrOutputParser()
    )
    return chain.invoke({"docset_name": name, "doc_fragment": doc_fragment})


def get_retrieval_tool_for_docset(
    docset_id: str, local_state: Dict[str, LocalIndexState]
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
        summarystore=local_state[docset_id].summaries_by_id,
        search_kwargs={"k": RETRIEVER_K},
        search_type=SearchType.mmr,
    )

    return create_retriever_tool(
        retriever,
        local_state[docset_id].retrieval_tool_function_name,
        local_state[docset_id].retrieval_tool_description,
    )
