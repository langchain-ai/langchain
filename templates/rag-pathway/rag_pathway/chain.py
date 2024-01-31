from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import PathwayVectorClient

from rag_pathway.server import run_vectorstoreserver

HOST = "127.0.0.1"
PORT = 8780

# If you have a running Pathway Vectorstore instance you can connect to it via
# client. If not, you can run an embedded Vectorstore as follows.
# Alternatively, you can run the vectorstore in another process.
create_vectorstore = True
if create_vectorstore:
    run_vectorstoreserver(host=HOST, port=PORT, threaded=True)

# Initalize client
client = PathwayVectorClient(
    host=HOST,
    port=PORT,
)

retriever = client.as_retriever()


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
