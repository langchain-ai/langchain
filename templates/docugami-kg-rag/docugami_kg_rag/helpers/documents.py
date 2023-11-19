from typing import Dict, List

from langchain.schema import Document, StrOutputParser
from langchain.prompts import ChatPromptTemplate
from tqdm import tqdm

from docugami_kg_rag.config import LLM, INCLUDE_XML_TAGS
from docugami_kg_rag.helpers.prompts import (
    ASSISTANT_SYSTEM_MESSAGE,
    CREATE_SUMMARY_PROMPT,
)


def get_parent_id_mappings(chunks: List[Document]) -> Dict[str, Document]:
    # Separate out unique child and parent chunks for small-to-big retrieval.

    parents: Dict[str, Document] = {}
    for chunk in chunks:
        if not chunk.parent:
            continue

        parent_id: str = chunk.parent.metadata["id"]

        # Set parent metadata on all child chunks
        chunk.metadata["doc_id"] = parent_id

        # Keep track of all unique parents (by ID)
        if parent_id not in parents:
            parents[parent_id] = chunk.parent

    return parents


def get_summary_mappings(docs_by_id: Dict[str, Document]) -> Dict[str, str]:
    # build summaries for all the given documents

    summaries: Dict[str, str] = {}
    format = "text" if not INCLUDE_XML_TAGS else "semantic XML without any namespaces or attributes"
    for id, doc in tqdm(docs_by_id.items(), "Creating summaries"):
        doc_fragment = doc.page_content[: 2048 * 10]  # Up to approximately 10 pages

        chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", ASSISTANT_SYSTEM_MESSAGE),
                    ("human", CREATE_SUMMARY_PROMPT),
                ]
            )
            | LLM
            | StrOutputParser()
        )
        summary = chain.invoke({"format": format, "doc_fragment": doc_fragment})
        summaries[id] = summary

    return summaries
