import os
from operator import itemgetter
from typing import List, Tuple

from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores.zep import CollectionConfig, ZepVectorStore
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    ConfigurableField,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables.utils import ConfigurableFieldSingleOption

ZEP_API_URL = os.environ.get("ZEP_API_URL", "http://localhost:8000")
ZEP_API_KEY = os.environ.get("ZEP_API_KEY", None)
ZEP_COLLECTION_NAME = os.environ.get("ZEP_COLLECTION", "langchaintest")

collection_config = CollectionConfig(
    name=ZEP_COLLECTION_NAME,
    description="Zep collection for LangChain",
    metadata={},
    embedding_dimensions=1536,
    is_auto_embedded=True,
)

vectorstore = ZepVectorStore(
    collection_name=ZEP_COLLECTION_NAME,
    config=collection_config,
    api_url=ZEP_API_URL,
    api_key=ZEP_API_KEY,
    embedding=None,
)

# Zep offers native, hardware-accelerated MMR. Enabling this will improve
# the diversity of results, but may also reduce relevance. You can tune
# the lambda parameter to control the tradeoff between relevance and diversity.
# Enabling is a good default.
retriever = vectorstore.as_retriever().configurable_fields(
    search_type=ConfigurableFieldSingleOption(
        id="search_type",
        options={"Similarity": "similarity", "Similarity with MMR Reranking": "mmr"},
        default="mmr",
        name="Search Type",
        description="Type of search to perform: 'similarity' or 'mmr'",
    ),
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search kwargs",
        description=(
            "Specify 'k' for number of results to return and 'lambda_mult' for tuning"
            " MMR relevance vs diversity."
        ),
    ),
)

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
    docs: List[Document],
    document_prompt: PromptTemplate = DEFAULT_DOCUMENT_PROMPT,
    document_separator: str = "\n\n",
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List[BaseMessage]:
    buffer: List[BaseMessage] = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


_condense_chain = (
    RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),
        # Condense follow-up question and chat into a standalone_question
        _condense_chain,
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(itemgetter("question")),
)


# User input
class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str


_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": _search_query | retriever | _combine_documents,
    }
).with_types(input_type=ChatHistory)

chain = _inputs | ANSWER_PROMPT | ChatOpenAI() | StrOutputParser()
