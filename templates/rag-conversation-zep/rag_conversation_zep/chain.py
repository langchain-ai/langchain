import os

from langchain.chat_models import ChatOpenAI
from langchain.memory.chat_message_histories import ZepChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

ZEP_API_URL = os.environ.get("ZEP_API_URL", "http://localhost:8000")

# RAG answer synthesis prompt
template = """Answer the question to the best of your knowledge:
"""
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)
_inputs = QA_PROMPT | ChatOpenAI() | StrOutputParser()


chain = RunnableWithMessageHistory(
    _inputs,
    lambda session_id: ZepChatMessageHistory(session_id, url=ZEP_API_URL),
    input_messages_key="question",
    history_messages_key="chat_history",
)
