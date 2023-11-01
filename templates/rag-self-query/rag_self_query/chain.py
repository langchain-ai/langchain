import os
from operator import itemgetter
from typing import List, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import SelfQueryRetriever
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import ConfigurableField, RunnablePassthrough
from langchain.vectorstores import ElasticsearchStore
from pydantic import BaseModel, Field

from .connection import es_connection_details
from .prompts import CONDENSE_QUESTION_PROMPT, DOCUMENT_PROMPT, LLM_CONTEXT_PROMPT

llm = ChatOpenAI(temperature=0)

DOCUMENT_CONTENTS = ""
METADATA_FIELD_INFO = {}


def retriever_from_vecstore(vecstore):
    return SelfQueryRetriever.from_llm(
        llm, vecstore, DOCUMENT_CONTENTS, METADATA_FIELD_INFO
    )


def _init_chroma():
    from langchain.vectorstores import Chroma

    vecstore = Chroma(
        collection_name=os.environ.get("CHROMA_COLLECTION_NAME"),
        embedding_function=OpenAIEmbeddings(),
    )
    return retriever_from_vecstore(vecstore)


def _init_redis():
    from langchain.vectorstores import Redis

    index_name = os.environ["REDIS_INDEX_NAME"]
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    return Redis(redis_url, index_name, OpenAIEmbeddings())


def _init_pinecone():
    from langchain.vectorstores import Pinecone

    index_name = os.environ["PINECONE_INDEX_NAME"]
    return Pinecone.from_existing_index(index_name, OpenAIEmbeddings())


def _init_supabase():
    from langchain.vectorstores import SupabaseVectorStore
    from supabase.client import create_client

    supabase_client = create_client(
        os.environ["SUPABASE_URL"], os.getenv("SUPABASE_KEY"), OpenAIEmbeddings()
    )
    return SupabaseVectorStore(
        supabase_client, OpenAIEmbeddings(), os.getenv("SUPABASE_TABLE_NAME")
    )


def _init_timescale():
    from langchain.vectorstores import TimescaleVector

    return TimescaleVector(
        os.environ["TIMESCALE_SERVICE_URL"],
        OpenAIEmbeddings(),
        os.getenv("TIMESCALE_COLLECTION_NAME"),
    )


def _init_weaviate():
    import weaviate
    from langchain.vectorstores import Weaviate

    client = weaviate.Client(
        url=os.environ["WEAVIATE_URL"], api_key=os.getenv("WEAVIATE_API_KEY")
    )
    return Weaviate(
        client, os.getenv("WEAVIATE_INDEX_NAME"), os.getenv("WEAVIATE_TEXT_KEY", "text")
    )


elastic = ElasticsearchStore(
    **es_connection_details,
    embedding=OpenAIEmbeddings(),
    index_name="workplace-search-example",
)
retriever = retriever_from_vecstore(elastic).configurable_alternatives(
    ConfigurableField(id="retriever"),
    chroma=_init_chroma,
    redis=_init_redis,
    pinecone=_init_pinecone,
    supabase=_init_supabase,
    timescale=_init_timescale,
    weaviate=_init_weaviate,
)


def _combine_documents(docs: List) -> str:
    return "\n\n".join(format_document(doc, prompt=DOCUMENT_PROMPT) for doc in docs)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    return "\n".join(f"Human: {human}\nAssistant: {ai}" for human, ai in chat_history)


class InputType(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = Field(default_factory=list)


standalone_question = (
    {
        "question": itemgetter("question"),
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser()
).with_types(input_type=InputType)

_context = {
    "context": retriever | _combine_documents,
    "question": RunnablePassthrough(),
}

chain = standalone_question | _context | LLM_CONTEXT_PROMPT | llm | StrOutputParser()
