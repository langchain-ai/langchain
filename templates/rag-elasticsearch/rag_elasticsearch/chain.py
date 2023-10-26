from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.schema import format_document
from typing import Tuple, List
from operator import itemgetter
from .prompts import CONDENSE_QUESTION_PROMPT, LLM_CONTEXT_PROMPT, DOCUMENT_PROMPT
from .connection import es_connection_details

# Setup connecting to Elasticsearch
vectorstore = ElasticsearchStore(
    **es_connection_details,
    embedding=HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    ),
    index_name="workplace-search-example",
)
retriever = vectorstore.as_retriever()

# Set up LLM to user
llm = ChatOpenAI(temperature=0)


def _combine_documents(
    docs, document_prompt=DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
)

_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}

chain = _inputs | _context | LLM_CONTEXT_PROMPT | llm | StrOutputParser()
