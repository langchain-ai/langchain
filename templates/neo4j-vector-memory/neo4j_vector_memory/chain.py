from operator import itemgetter

from langchain_community.vectorstores import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from neo4j_vector_memory.history import get_history, save_history

# Define vector retrieval
retrieval_query = "RETURN node.text AS text, score, {id:elementId(node)} AS metadata"
vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(), index_name="dune", retrieval_query=retrieval_query
)
retriever = vectorstore.as_retriever()

# Define LLM
llm = ChatOpenAI()


# Condense a chat history and follow-up question into a standalone question
condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Make sure to include all the relevant information.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

# RAG answer synthesis prompt
answer_template = """Answer the question based only on the following context:
<context>
{context}
</context>"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", answer_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

chain = (
    RunnablePassthrough.assign(chat_history=get_history)
    | RunnablePassthrough.assign(
        rephrased_question=CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    )
    | RunnablePassthrough.assign(
        context=itemgetter("rephrased_question") | retriever,
    )
    | RunnablePassthrough.assign(
        output=ANSWER_PROMPT | llm | StrOutputParser(),
    )
    | save_history
)


# Add typing for input
class Question(BaseModel):
    question: str
    user_id: str
    session_id: str


chain = chain.with_types(input_type=Question)
