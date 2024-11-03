from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import ConfigurableField, RunnableParallel
from langchain_openai import ChatOpenAI

from neo4j_advanced_rag.retrievers import (
    hypothetic_question_vectorstore,
    parent_vectorstore,
    summary_vectorstore,
    typical_rag,
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

retriever = typical_rag.as_retriever().configurable_alternatives(
    ConfigurableField(id="strategy"),
    default_key="typical_rag",
    parent_strategy=parent_vectorstore.as_retriever(),
    hypothetical_questions=hypothetic_question_vectorstore.as_retriever(),
    summary_strategy=summary_vectorstore.as_retriever(),
)

chain = (
    RunnableParallel(
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
    )
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    question: str


chain = chain.with_types(input_type=Question)
