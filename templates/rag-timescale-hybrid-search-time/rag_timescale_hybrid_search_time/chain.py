# ruff: noqa: E501

import os
from datetime import timedelta

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.vectorstores.timescalevector import TimescaleVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from .load_sample_dataset import load_ts_git_dataset

# to enable debug uncomment the following lines:
# from langchain.globals import set_debug
# set_debug(True)

# from dotenv import find_dotenv, load_dotenv
# _ = load_dotenv(find_dotenv())

if os.environ.get("TIMESCALE_SERVICE_URL", None) is None:
    raise Exception("Missing `TIMESCALE_SERVICE_URL` environment variable.")

SERVICE_URL = os.environ["TIMESCALE_SERVICE_URL"]
LOAD_SAMPLE_DATA = os.environ.get("LOAD_SAMPLE_DATA", False)


# DATASET SPECIFIC CODE
# Load the sample dataset. You will have to change this to load your own dataset.
collection_name = "timescale_commits"
partition_interval = timedelta(days=7)
if LOAD_SAMPLE_DATA:
    load_ts_git_dataset(
        SERVICE_URL,
        collection_name=collection_name,
        num_records=500,
        partition_interval=partition_interval,
    )

# This will change depending on the metadata stored in your dataset.
document_content_description = "The git log commit summary containing the commit hash, author, date of commit, change summary and change details"
metadata_field_info = [
    AttributeInfo(
        name="id",
        description="A UUID v1 generated from the date of the commit",
        type="uuid",
    ),
    AttributeInfo(
        # This is a special attribute represent the timestamp of the uuid.
        name="__uuid_timestamp",
        description="The timestamp of the commit. Specify in YYYY-MM-DDTHH::MM:SSZ format",
        type="datetime.datetime",
    ),
    AttributeInfo(
        name="author_name",
        description="The name of the author of the commit",
        type="string",
    ),
    AttributeInfo(
        name="author_email",
        description="The email address of the author of the commit",
        type="string",
    ),
]
# END DATASET SPECIFIC CODE

embeddings = OpenAIEmbeddings()
vectorstore = TimescaleVector(
    embedding=embeddings,
    collection_name=collection_name,
    service_url=SERVICE_URL,
    time_partition_interval=partition_interval,
)

llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True,
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(temperature=0, model="gpt-4")

# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
