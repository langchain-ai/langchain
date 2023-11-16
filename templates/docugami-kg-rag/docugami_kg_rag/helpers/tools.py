from typing import Dict
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.tools.base import BaseTool
from langchain.vectorstores.pinecone import Pinecone


from docugami_kg_rag.config import EMBEDDINGS, PINECONE_INDEX, RETRIEVER_K, LocalIndexState


def build_retrieval_tool_for_docset(docset_id: str, local_state: Dict[str, LocalIndexState]) -> BaseTool:
    # Chunks are in the vector store, and full documents are in the store inside the local state

    docset_pinecone_index_name = f"{PINECONE_INDEX}-{docset_id}"
    chunk_vectorstore = Pinecone.from_existing_index(docset_pinecone_index_name, EMBEDDINGS)
    retriever = MultiVectorRetriever(
        vectorstore=chunk_vectorstore,
        docstore=local_state[docset_id].parents_by_id,
        search_kwargs={"k": RETRIEVER_K},
    )

    return create_retriever_tool(
        retriever,
        local_state[docset_id].retrieval_tool_function_name,
        local_state[docset_id].retrieval_tool_description,
    )
