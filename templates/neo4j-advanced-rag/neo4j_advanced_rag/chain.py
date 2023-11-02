from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from langchain.schema.runnable import ConfigurableField

from operator import itemgetter

from neo4j_advanced_rag.retrievers import (
    summary_vectorstore,
    parent_vectorstore,
    hypothetic_question_vectorstore,
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

retriever = summary_vectorstore.as_retriever().configurable_alternatives(
    ConfigurableField(id="strategy"),
    parent_document=parent_vectorstore.as_retriever(),
    hypothetical_questions=hypothetic_question_vectorstore.as_retriever(),
)

chain = (
    RunnableParallel(
        {
            "context": itemgetter("question") | retriever,
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
