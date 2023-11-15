import os

from langchain.llms.bedrock import Bedrock
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.retrievers import AmazonKendraRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

# Get region and profile from env
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
profile = os.environ.get("AWS_PROFILE", "default")
kendra_index = os.environ.get("KENDRA_INDEX_ID", None)

if not kendra_index:
    raise ValueError(
        "No value provided in env variable 'KENDRA_INDEX_ID'. "
        "A Kendra index is required to run this application."
    )

# Set LLM and embeddings
model = Bedrock(
    model_id="anthropic.claude-v2",
    region_name=region,
    credentials_profile_name=profile,
    model_kwargs={"max_tokens_to_sample": 200},
)

# Create Kendra retriever
retriever = AmazonKendraRetriever(index_id=kendra_index, top_k=5, region_name=region)

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
