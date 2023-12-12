from langchain.chat_models import ChatCohere
from langchain.retrievers import CohereRagRetriever

rag = CohereRagRetriever(llm=ChatCohere())


def get_docs_message(message):
    docs = rag.get_relevant_documents(message)
    message_doc = next(
        (x for x in docs if x.metadata.get("type") == "model_response"), None
    )
    return message_doc.page_content


def librarian_rag(x):
    return get_docs_message(x["message"])
