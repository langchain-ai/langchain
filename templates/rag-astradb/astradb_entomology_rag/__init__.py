import os

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import AstraDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .populate_vector_store import populate

# inits
llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()
vector_store = AstraDB(
    embedding=embeddings,
    collection_name="langserve_rag_demo",
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
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
