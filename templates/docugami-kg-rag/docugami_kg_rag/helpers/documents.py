from typing import Dict, List

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document, StrOutputParser
from tqdm import tqdm

from docugami_kg_rag.config import INCLUDE_XML_TAGS, LARGE_FRAGMENT_MAX_TEXT_LENGTH, LLM
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


def build_summary_mappings(docs_by_id: Dict[str, Document]) -> Dict[str, str]:
    # build summaries for all the given documents

    summaries: Dict[str, str] = {}
    format = (
        "text"
        if not INCLUDE_XML_TAGS
        else "semantic XML without any namespaces or attributes"
    )

    for id, doc in tqdm(docs_by_id.items(), "Creating full document summaries"):
        doc_fragment = doc.page_content[:LARGE_FRAGMENT_MAX_TEXT_LENGTH]

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
