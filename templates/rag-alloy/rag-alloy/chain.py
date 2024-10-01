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
from typing import Optional

from config import (
    CLUSTER,
    DATABASE,
    INSTANCE,
    PASSWORD,
    PROJECT_ID,
    REGION,
    STAGING_BUCKET,
    TABLE_NAME,
    USER,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from pydantic import BaseModel
from vertexai.preview import reasoning_engines  # type: ignore

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore

# This sample requires a vector store table
# Create these tables using `AlloyDBEngine` method `init_vectorstore_table()`
# or create and load the table using `create_embeddings.py`


class AlloyDBRetriever(reasoning_engines.Queryable):
    def __init__(
        self,
        model: str,
        project: str,
        region: str,
        cluster: str,
        instance: str,
        database: str,
        table: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.model_name = model
        self.project = project
        self.region = region
        self.cluster = cluster
        self.instance = instance
        self.database = database
        self.table = table
        self.user = user
        self.password = password

    def set_up(self):
        """All unpickle-able logic should go here.
        In general, add any logic that requires a network or database
        connection.
        """
        # Create a chain to handle the processing of relevant documents
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
                ("human", "{input}"),
            ]
        )
        llm = VertexAI(model_name=self.model_name, project=self.project)
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)

        # Initialize the vector store and retriever
        engine = AlloyDBEngine.from_instance(
            self.project,
            self.region,
            self.cluster,
            self.instance,
            self.database,
            user=self.user,
            password=self.password,
        )
        vector_store = AlloyDBVectorStore.create_sync(
            engine,
            table_name=self.table,
            embedding_service=VertexAIEmbeddings(
                model_name="textembedding-gecko@latest", project=self.project
            ),
        )
        retriever = vector_store.as_retriever()

        # Create a retrieval chain to fetch relevant documents and pass them to
        # an LLM to generate a response
        self.chain = create_retrieval_chain(retriever, combine_docs_chain)

    def query(self, input: str) -> str:
        """Query the application.

        Args:
            input: The user query.

        Returns:
            The LLM response dictionary.
        """
        # Define the runtime logic that serves user queries
        response = self.chain.invoke({"input": input})
        return response["answer"]

app = AlloyDBRetriever(
    model="gemini-pro",
    project=PROJECT_ID,
    region=REGION,
    cluster=CLUSTER,
    instance=INSTANCE,
    database=DATABASE,
    table=TABLE_NAME,
    user=USER,
    password=PASSWORD,
)
app.set_up()
chain = app.chain

