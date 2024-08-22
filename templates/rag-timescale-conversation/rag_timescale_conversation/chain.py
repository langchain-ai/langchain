import os
from datetime import datetime, timedelta
from operator import itemgetter
from typing import List, Optional, Tuple

from dotenv import find_dotenv, load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.timescalevector import TimescaleVector
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from .load_sample_dataset import load_ts_git_dataset

load_dotenv(find_dotenv())

if os.environ.get("TIMESCALE_SERVICE_URL", None) is None:
    raise Exception("Missing `TIMESCALE_SERVICE_URL` environment variable.")

SERVICE_URL = os.environ["TIMESCALE_SERVICE_URL"]
LOAD_SAMPLE_DATA = os.environ.get("LOAD_SAMPLE_DATA", False)
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "timescale_commits")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4")

partition_interval = timedelta(days=7)
if LOAD_SAMPLE_DATA:
    load_ts_git_dataset(
        SERVICE_URL,
        collection_name=COLLECTION_NAME,
        num_records=500,
        partition_interval=partition_interval,
    )

embeddings = OpenAIEmbeddings()
vectorstore = TimescaleVector(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    service_url=SERVICE_URL,
    time_partition_interval=partition_interval,
)
retriever = vectorstore.as_retriever()

# Condense a chat history and follow-up question into a standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# RAG answer synthesis prompt
template = """Answer the question based only on the following context:
<context>
{context}
</context>"""
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

# Conversational Retrieval Chain
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


# User input
class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    metadata_filter: Optional[dict]


_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            retriever_query=RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | ChatOpenAI(temperature=0, model=OPENAI_MODEL)
            | StrOutputParser()
        ),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnablePassthrough.assign(retriever_query=lambda x: x["question"]),
)


def get_retriever_with_metadata(x):
    start_dt = x.get("start_date", None)
    end_dt = x.get("end_date", None)
    metadata_filter = x.get("metadata_filter", None)
    opt = {}

    if start_dt is not None:
        opt["start_date"] = start_dt
    if end_dt is not None:
        opt["end_date"] = end_dt
    if metadata_filter is not None:
        opt["filter"] = metadata_filter
    v = vectorstore.as_retriever(search_kwargs=opt)
    return RunnableLambda(itemgetter("retriever_query")) | v


_retriever = RunnableLambda(get_retriever_with_metadata)

_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "start_date": lambda x: x.get("start_date", None),
        "end_date": lambda x: x.get("end_date", None),
        "context": _search_query | _retriever | _combine_documents,
    }
)

_datetime_to_string = RunnablePassthrough.assign(
    start_date=lambda x: (
        x.get("start_date", None).isoformat()
        if x.get("start_date", None) is not None
        else None
    ),
    end_date=lambda x: (
        x.get("end_date", None).isoformat()
        if x.get("end_date", None) is not None
        else None
    ),
).with_types(input_type=ChatHistory)

chain = (
    _datetime_to_string
    | _inputs
    | ANSWER_PROMPT
    | ChatOpenAI(model=OPENAI_MODEL)
    | StrOutputParser()
)
