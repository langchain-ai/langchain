from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import Neo4jVector

retrieval_query = """
MATCH (node)-[:HAS_PARENT]->(parent)
RETURN parent.text AS text, score, {} AS metadata
"""

vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    index_name="retrieval",
    node_label="Child",
    embedding_node_property="embedding",
    retrieval_query=retrieval_query

)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)