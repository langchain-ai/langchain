from langchain.callbacks import streaming_stdout
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever


# Make this look better in the docs.
class Question(BaseModel):
    __root__: str


# Init Embeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

knowledge_base = Chroma(
    persist_directory="/tmp/xeon_rag_db",
    embedding_function=embedder,
    collection_name="xeon-rag",
)
query = "What was Nike's revenue in 2023?"
docs = knowledge_base.similarity_search(query)
print(docs[0].page_content)
retriever = VectorStoreRetriever(
    vectorstore=knowledge_base, search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
)

# Define our prompt
template = """
Use the following pieces of context from retrieved
dataset to answer the question. Do not make up an answer if there is no
context provided to help answer it.

Context:
---------
{context}

---------
Question: {question}
---------

Answer:
"""


prompt = ChatPromptTemplate.from_template(template)


ENDPOINT_URL = "http://localhost:8080"
callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
model = HuggingFaceEndpoint(
    endpoint_url=ENDPOINT_URL,
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    streaming=True,
)

# RAG Chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
).with_types(input_type=Question)
