from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from .chat import chat
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.retrievers import CohereRagRetriever
from langchain.chat_models import ChatCohere
from langchain.chains.question_answering import load_qa_chain
from .chat import chat


rag = CohereRagRetriever(llm=ChatCohere())



def get_docs_message(message):
    docs = rag.get_relevant_documents(message)
    message_doc = next((x for x in docs if x.metadata.get("type") == "model_response"), None)
    return message_doc.page_content

librarian_rag = lambda x: get_docs_message(x["message"])