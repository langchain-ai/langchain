import os

import cassio
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .populate_vector_store import populate

use_cassandra = int(os.environ.get("USE_CASSANDRA_CLUSTER", "0"))
if use_cassandra:
    from .cassandra_cluster_init import get_cassandra_connection

    session, keyspace = get_cassandra_connection()
    cassio.init(
        session=session,
        keyspace=keyspace,
    )
else:
    cassio.init(
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        database_id=os.environ["ASTRA_DB_ID"],
        keyspace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )


# inits
llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()
vector_store = Cassandra(
    session=None,
    keyspace=None,
    embedding=embeddings,
    table_name="langserve_rag_demo",
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# For demo reasons, let's ensure there are rows on the vector store.
# Please remove this and/or adapt to your use case!
inserted_lines = populate(vector_store)
if inserted_lines:
    print(f"Done ({inserted_lines} lines inserted).")  # noqa: T201

entomology_template = """
You are an expert entomologist, tasked with answering enthusiast biologists' questions.
You must answer based only on the provided context, do not make up any fact.
Your answers must be concise and to the point, but strive to provide scientific details
(such as family, order, Latin names, and so on when appropriate).
You MUST refuse to answer questions on other topics than entomology,
as well as questions whose answer is not found in the provided context.

CONTEXT:
{context}

QUESTION: {question}

YOUR ANSWER:"""

entomology_prompt = ChatPromptTemplate.from_template(entomology_template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | entomology_prompt
    | llm
    | StrOutputParser()
)
