import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import Vectara

if os.environ.get("VECTARA_CUSTOMER_ID", None) is None:
    raise Exception("Missing `VECTARA_CUSTOMER_ID` environment variable.")
if os.environ.get("VECTARA_CORPUS_ID", None) is None:
    raise Exception("Missing `VECTARA_CORPUS_ID` environment variable.")
if os.environ.get("VECTARA_API_KEY", None) is None:
    raise Exception("Missing `VECTARA_API_KEY` environment variable.")

# If you want to ingest data then use this code.
# Note that no document chunking is needed, as this is
# done efficiently in the Vectara backend.
#   loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
#   docs = loader.load()
#   vec_store = Vectara.from_document(docs)
#   retriever = vec_store.as_retriever()
# Otherwise, if data is already loaded into Vectara then use this code:
retriever = Vectara().as_retriever()

# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = ChatOpenAI()
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
