import os

from langchain.retrievers import GoogleVertexAISearchRetriever
from langchain_community.chat_models import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Get project, data store, and model type from env variables
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
data_store_id = os.environ.get("DATA_STORE_ID")
model_type = os.environ.get("MODEL_TYPE")

if not data_store_id:
    raise ValueError(
        "No value provided in env variable 'DATA_STORE_ID'. "
        "A  data store is required to run this application."
    )

# Set LLM and embeddings
model = ChatVertexAI(model_name=model_type, temperature=0.0)

# Create Vertex AI retriever
retriever = GoogleVertexAISearchRetriever(
    project_id=project_id, search_engine_id=data_store_id
)

# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
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
