import os

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import FAISS

# Get region and profile from env
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
profile = os.environ.get("AWS_PROFILE", "default")

# Set LLM and embeddings
model = Bedrock(
    model_id="anthropic.claude-v2",
    region_name=region,
    credentials_profile_name=profile,
    model_kwargs={"max_tokens_to_sample": 200},
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")

# Add to vectorDB
vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=bedrock_embeddings
)
retriever = vectorstore.as_retriever()

# Get retriever from vectorstore
retriever = vectorstore.as_retriever()

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
