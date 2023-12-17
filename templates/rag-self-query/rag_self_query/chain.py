import os
from operator import itemgetter
from typing import List, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import SelfQueryRetriever
from langchain.schema import format_document
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from .prompts import CONDENSE_QUESTION_PROMPT, DOCUMENT_PROMPT, LLM_CONTEXT_PROMPT

ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME", "elastic")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ELASTIC_INDEX_NAME = os.getenv("ELASTIC_INDEX_NAME", "workspace-search-example")

if ELASTIC_CLOUD_ID and ELASTIC_USERNAME and ELASTIC_PASSWORD:
    es_connection_details = {
        "es_cloud_id": ELASTIC_CLOUD_ID,
        "es_user": ELASTIC_USERNAME,
        "es_password": ELASTIC_PASSWORD,
    }
else:
    es_connection_details = {"es_url": ES_URL}

vecstore = ElasticsearchStore(
    ELASTIC_INDEX_NAME,
    embedding=OpenAIEmbeddings(),
    **es_connection_details,
)

document_contents = "The purpose and specifications of a workplace policy."
metadata_field_info = [
    {"name": "name", "type": "string", "description": "Name of the workplace policy."},
    {
        "name": "created_on",
        "type": "date",
        "description": "The date the policy was created in ISO 8601 date format (YYYY-MM-DD).",  # noqa: E501
    },
    {
        "name": "updated_at",
        "type": "date",
        "description": "The date the policy was last updated in ISO 8601 date format (YYYY-MM-DD).",  # noqa: E501
    },
    {
        "name": "location",
        "type": "string",
        "description": "Where the policy text is stored. The only valid values are ['github', 'sharepoint'].",  # noqa: E501
    },
]
llm = ChatOpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm, vecstore, document_contents, metadata_field_info
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
)


def route_question(input):
    if input.get("chat_history"):
        return standalone_question
    else:
        return RunnablePassthrough()


_context = RunnableParallel(
    context=retriever | _combine_documents,
    question=RunnablePassthrough(),
)


chain = (
    standalone_question | _context | LLM_CONTEXT_PROMPT | llm | StrOutputParser()
).with_types(input_type=InputType)
