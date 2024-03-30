from langchain_community.chat_models import ChatOpenAI
from langchain_core.load import load
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnablePassthrough

from propositional_retrieval.constants import DOCSTORE_ID_KEY
from propositional_retrieval.storage import get_multi_vector_retriever


def format_docs(docs: list) -> str:
    loaded_docs = [load(doc) for doc in docs]
    return "\n".join(
        [
            f"<Document id={i}>\n{doc.page_content}\n</Document>"
            for i, doc in enumerate(loaded_docs)
        ]
    )


def rag_chain(retriever):
    """
    The RAG chain

    :param retriever: A function that retrieves the necessary context for the model.
    :return: A chain of functions representing the multi-modal RAG process.
    """
    model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", max_tokens=1024)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant. Answer based on the retrieved documents:"
                "\n<Documents>\n{context}\n</Documents>",
            ),
            ("user", "{question}?"),
        ]
    )

    # Define the RAG pipeline
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain


# Create the multi-vector retriever
retriever = get_multi_vector_retriever(DOCSTORE_ID_KEY)

# Create RAG chain
chain = rag_chain(retriever)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
