{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "fcc8bb1c",
   "metadata": {},
   "source": [
    "# Getting Started\n",
    "\n",
    "LangChain primary focuses on constructing indexes with the goal of using them as a Retriever. In order to best understand what this means, it's worth highlighting what the base Retriever interface is. The `BaseRetriever` class in LangChain is as follows:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b09ac324",
   "metadata": {},
   "outputs": [],
   "source": [
    "from abc import ABC, abstractmethod\n",
    "from typing import List\n",
    "from langchain.schema import Document\n",
    "\n",
    "class BaseRetriever(ABC):\n",
    "    @abstractmethod\n",
    "    def get_relevant_documents(self, query: str) -> List[Document]:\n",
    "        \"\"\"Get texts relevant for a query.\n",
    "\n",
    "        Args:\n",
    "            query: string to find relevant texts for\n",
    "\n",
    "        Returns:\n",
    "            List of relevant documents\n",
    "        \"\"\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e19d4adb",
   "metadata": {},
   "source": [
    "It's that simple! The `get_relevant_documents` method can be implemented however you see fit.\n",
    "\n",
    "Of course, we also help construct what we think useful Retrievers are. The main type of Retriever that we focus on is a Vectorstore retriever. We will focus on that for the rest of this guide.\n",
    "\n",
    "In order to understand what a vectorstore retriever is, it's important to understand what a Vectorstore is. So let's look at that."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2244801b",
   "metadata": {},
   "source": [
    "By default, LangChain uses [Chroma](../../ecosystem/chroma.md) as the vectorstore to index and search embeddings. To walk through this tutorial, we'll first need to install `chromadb`.\n",
    "\n",
    "```\n",
    "pip install chromadb\n",
    "```\n",
    "\n",
    "This example showcases question answering over documents.\n",
    "We have chosen this as the example for getting started because it nicely combines a lot of different elements (Text splitters, embeddings, vectorstores) and then also shows how to use them in a chain.\n",
    "\n",
    "Question answering over documents consists of four steps:\n",
    "\n",
    "1. Create an index\n",
    "2. Create a Retriever from that index\n",
    "3. Create a question answering chain\n",
    "4. Ask questions!\n",
    "\n",
    "Each of the steps has multiple sub steps and potential configurations. In this notebook we will primarily focus on (1). We will start by showing the one-liner for doing so, but then break down what is actually going on.\n",
    "\n",
    "First, let's import some common classes we'll use no matter what."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8d369452",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.chains import RetrievalQA\n",
    "from langchain.llms import OpenAI"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "07c1e3b9",
   "metadata": {},
   "source": [
    "Next in the generic setup, let's specify the document loader we want to use. You can download the `state_of_the_union.txt` file [here](https://github.com/hwchase17/langchain/blob/master/docs/modules/state_of_the_union.txt)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "33958a86",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.document_loaders import TextLoader\n",
    "loader = TextLoader('../state_of_the_union.txt', encoding='utf8')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "489c74bb",
   "metadata": {},
   "source": [
    "## One Line Index Creation\n",
    "\n",
    "To get started as quickly as possible, we can use the `VectorstoreIndexCreator`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "403fc231",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.indexes import VectorstoreIndexCreator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "57a8a199",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running Chroma using direct local API.\n",
      "Using DuckDB in-memory for database. Data will be transient.\n"
     ]
    }
   ],
   "source": [
    "index = VectorstoreIndexCreator().from_loaders([loader])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f3493fa4",
   "metadata": {},
   "source": [
    "Now that the index is created, we can use it to ask questions of the data! Note that under the hood this is actually doing a few steps as well, which we will cover later in this guide."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "23d0d234",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\" The president said that Ketanji Brown Jackson is one of the nation's top legal minds, a former top litigator in private practice, a former federal public defender, and from a family of public school educators and police officers. He also said that she is a consensus builder and has received a broad range of support from the Fraternal Order of Police to former judges appointed by Democrats and Republicans.\""
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "query = \"What did the president say about Ketanji Brown Jackson\"\n",
    "index.query(query)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ae46b239",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'question': 'What did the president say about Ketanji Brown Jackson',\n",
       " 'answer': \" The president said that he nominated Circuit Court of Appeals Judge Ketanji Brown Jackson, one of the nation's top legal minds, to continue Justice Breyer's legacy of excellence, and that she has received a broad range of support from the Fraternal Order of Police to former judges appointed by Democrats and Republicans.\\n\",\n",
       " 'sources': '../state_of_the_union.txt'}"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "query = \"What did the president say about Ketanji Brown Jackson\"\n",
    "index.query_with_sources(query)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ff100212",
   "metadata": {},
   "source": [
    "What is returned from the `VectorstoreIndexCreator` is `VectorStoreIndexWrapper`, which provides these nice `query` and `query_with_sources` functionality. If we just wanted to access the vectorstore directly, we can also do that."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b04f3c10",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<langchain.vectorstores.chroma.Chroma at 0x119aa5940>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "index.vectorstore"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "297ccfa4",
   "metadata": {},
   "source": [
    "If we then want to access the VectorstoreRetriever, we can do that with:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "b8fef77d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "VectorStoreRetriever(vectorstore=<langchain.vectorstores.chroma.Chroma object at 0x119aa5940>, search_kwargs={})"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "index.vectorstore.as_retriever()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2cb6d2eb",
   "metadata": {},
   "source": [
    "## Walkthrough\n",
    "\n",
    "Okay, so what's actually going on? How is this index getting created?\n",
    "\n",
    "A lot of the magic is being hid in this `VectorstoreIndexCreator`. What is this doing?\n",
    "\n",
    "There are three main steps going on after the documents are loaded:\n",
    "\n",
    "1. Splitting documents into chunks\n",
    "2. Creating embeddings for each document\n",
    "3. Storing documents and embeddings in a vectorstore\n",
    "\n",
    "Let's walk through this in code"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "54270abc",
   "metadata": {},
   "outputs": [],
   "source": [
    "documents = loader.load()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9fdc0fc2",
   "metadata": {},
   "source": [
    "Next, we will split the documents into chunks."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "afecb8cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.text_splitter import CharacterTextSplitter\n",
    "text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)\n",
    "texts = text_splitter.split_documents(documents)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4bebc041",
   "metadata": {},
   "source": [
    "We will then select which embeddings we want to use."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "9eaaa735",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.embeddings import OpenAIEmbeddings\n",
    "embeddings = OpenAIEmbeddings()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "24612905",
   "metadata": {},
   "source": [
    "We now create the vectorstore to use as the index."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "5c7049db",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running Chroma using direct local API.\n",
      "Using DuckDB in-memory for database. Data will be transient.\n"
     ]
    }
   ],
   "source": [
    "from langchain.vectorstores import Chroma\n",
    "db = Chroma.from_documents(texts, embeddings)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f0ef85a6",
   "metadata": {},
   "source": [
    "So that's creating the index. Then, we expose this index in a retriever interface."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "13495c77",
   "metadata": {},
   "outputs": [],
   "source": [
    "retriever = db.as_retriever()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "30c4e5c6",
   "metadata": {},
   "source": [
    "Then, as before, we create a chain and use it to answer questions!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "3018f865",
   "metadata": {},
   "outputs": [],
   "source": [
    "qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type=\"stuff\", retriever=retriever)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "032a47f8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\" The President said that Judge Ketanji Brown Jackson is one of the nation's top legal minds, a former top litigator in private practice, a former federal public defender, and from a family of public school educators and police officers. He said she is a consensus builder and has received a broad range of support from organizations such as the Fraternal Order of Police and former judges appointed by Democrats and Republicans.\""
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "query = \"What did the president say about Ketanji Brown Jackson\"\n",
    "qa.run(query)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9464690e",
   "metadata": {},
   "source": [
    "`VectorstoreIndexCreator` is just a wrapper around all this logic. It is configurable in the text splitter it uses, the embeddings it uses, and the vectorstore it uses. For example, you can configure it as below:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "4001bbc6",
   "metadata": {},
   "outputs": [],
   "source": [
    "index_creator = VectorstoreIndexCreator(\n",
    "    vectorstore_cls=Chroma, \n",
    "    embedding=OpenAIEmbeddings(),\n",
    "    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "78d8d143",
   "metadata": {},
   "source": [
    "Hopefully this highlights what is going on under the hood of `VectorstoreIndexCreator`. While we think it's important to have a simple way to create indexes, we also think it's important to understand what's going on under the hood."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dd7257bf",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.1"
  },
  "vscode": {
   "interpreter": {
    "hash": "b1677b440931f40d89ef8be7bf03acb108ce003de0ac9b18e8d43753ea2e7103"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
