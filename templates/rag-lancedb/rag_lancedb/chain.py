import lancedb
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Example for document loading (from url), splitting, and creating vectostore
"""
# Load
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# embedding
embedding = OpenAIEmbeddings()

# Add to vectorDB
db = lancedb.connect("/tmp/lancedb")
table = db.create_table(
    "my_table",
    data=[
        {
            "vector": embeddings.embed_query("Hello World"),
            "text": "Hello World",
            "id": "1",
        }
    ],
    mode="overwrite",
)

vectorstore = LanceDB.from_documents(all_splits, embeddings, connection=table)

retriever = vectorstore.as_retriever()

"""

# Embedding model, DB setting, and embedding a single document
embedding = OpenAIEmbeddings()

db = lancedb.connect("/tmp/lancedb")
table = db.create_table(
    "my_table",
    data=[
        {
            "vector": embedding.embed_query("Hello World"),
            "text": "Hello World",
            "id": "1",
        }
    ],
    mode="overwrite",
)

vectorstore = LanceDB.from_texts(
    ["harrison worked at kensho"], embedding, connection=table
)
retriever = vectorstore.as_retriever()
# Template, LLM, RAG chain, and typing for input
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


class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
