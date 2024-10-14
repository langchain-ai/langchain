# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from pydantic import RootModel

# This sample requires a vector store table
# Create this table using the `AlloyDBEngine` method `init_vectorstore_table()`
# Learn more about setting up an `AlloyDBVectorStore` at
# https://github.com/googleapis/langchain-google-alloydb-pg-python/blob/main/docs/vector_store.ipynb


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v

PROJECT_ID = get_env_var("PROJECT_ID", "project id for google cloud")
REGION = get_env_var("REGION", "region for AlloyDB instance")
CLUSTER = get_env_var("CLUSTER_ID", "cluster for AlloyDB")
DATABASE = get_env_var("DATABASE_ID", "database name on AlloyDB instance")
INSTANCE = get_env_var("INSTANCE_ID", "instance for AlloyDB")
TABLE_NAME = get_env_var("TABLE_NAME", "table name on AlloyDB instance")
USER = get_env_var("DB_USER", "database user for AlloyDB")
PASSWORD = get_env_var("DB_PASSWORD", "database password for AlloyDB")

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)
llm = ChatVertexAI(model_name="gemini-pro", project=PROJECT_ID)

# Initialize the vector store and retriever
engine = AlloyDBEngine.from_instance(
    PROJECT_ID,
    REGION,
    CLUSTER,
    INSTANCE,
    DATABASE,
    USER,
    PASSWORD,
)
vector_store = AlloyDBVectorStore.create_sync(
    engine,
    table_name=TABLE_NAME,
    embedding_service=VertexAIEmbeddings(
        model_name="textembedding-gecko@latest", project=PROJECT_ID
    ),
)
retriever = vector_store.as_retriever()

# Create a retrieval chain to fetch relevant documents and pass them to
# an LLM to generate a response
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)


# Add typing for input
class Question(RootModel):
    root: str


chain = chain.with_types(input_type=Question)
