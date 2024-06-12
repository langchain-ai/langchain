
# # Embedding Documents using Optimized and Quantized Embedders
# 
# In this tutorial, we will demo how to build a RAG pipeline, with the embedding for all documents done using Quantized Embedders.
# 
# We will use a pipeline that will:
# 
# * Create a document collection.
# * Embed all documents using Quantized Embedders.
# * Fetch relevant documents for our question.
# * Run an LLM answer the question.
# 
# For more information about optimized models, we refer to [optimum-intel](https://github.com/huggingface/optimum-intel.git) and [IPEX](https://github.com/intel/intel-extension-for-pytorch).
# 
# This tutorial is based on the [Langchain RAG tutorial here](https://towardsai.net/p/machine-learning/dense-x-retrieval-technique-in-langchain-and-llamaindex).
# and [RAG with quantized embeddings cookbook here] (https://github.com/langchain-ai/langchain/blob/36cad5d25c411824c7de7aa828fe56fa6fbc02df/cookbook/rag_with_quantized_embeddings.ipynb)


import uuid
from pathlib import Path

import langchain
import torch
from bs4 import BeautifulSoup as Soup
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore, LocalFileStore
from langchain_community.document_loaders.recursive_url_loader import (
    RecursiveUrlLoader,
)
from langchain_community.vectorstores import Chroma

# For our example, we'll load docs from the web
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCSTORE_DIR = "."
DOCSTORE_ID_KEY = "doc_id"


# Lets first load up this paper, and split into text chunks of size 1000.
loader = RecursiveUrlLoader(
    "https://ar5iv.labs.arxiv.org/html/1706.03762",
    max_depth=2,
    extractor=lambda x: Soup(x, "html.parser").text,
)
data = loader.load()
print(f"Loaded {len(data)} documents")

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
print(f"Split into {len(all_splits)} documents")


# In order to embed our documents, we can use the ```QuantizedBiEncoderEmbeddings```, for efficient and fast embedding. 

from langchain_community.embeddings import QuantizedBiEncoderEmbeddings
from langchain_core.embeddings import Embeddings

model_name = "Intel/bge-small-en-v1.5-rag-int8-static"
encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity

model_inc = QuantizedBiEncoderEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs,
    query_instruction="Represent this sentence for searching relevant passages: ",
)

# With our embedder in place, lets define our retriever:

def get_multi_vector_retriever(
    docstore_id_key: str, collection_name: str, embedding_function: Embeddings
):
    """Create the composed retriever object."""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    store = InMemoryByteStore()

    return MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=docstore_id_key,
    )


retriever = get_multi_vector_retriever(DOCSTORE_ID_KEY, "multi_vec_store", model_inc)


# Next, we divide each chunk into sub-docs:

child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
id_key = "doc_id"
doc_ids = [str(uuid.uuid4()) for _ in all_splits]


sub_docs = []
for i, doc in enumerate(all_splits):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)

# Lets write our documents into our new store. This will use our embedder on each document.

retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, all_splits)))

# Great! Our retriever is good to go. Lets load up an LLM, that will reason over the retrieved documents:


import sys
workspace="/home/rbrugaro/langchain/libs/partners/huggingface"
sys.path.insert(0, workspace)
import torch
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


hf = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-2-7b-hf",
    task="text-generation",
    backend="ipex",
    pipeline_kwargs={"max_new_tokens": 100},
)

# Next, we will load up a prompt for answering questions using retrieved documents:

from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

# We can now build our pipeline:

from langchain.schema.runnable import RunnablePassthrough

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | hf


# Excellent! lets ask it a question.
# We will also use a verbose and debug, to check which documents were used by the model to produce the answer.


langchain.verbose = True
langchain.debug = True

llm_res = rag_chain.invoke(
    "What is the first transduction model relying entirely on self-attention?",
)



