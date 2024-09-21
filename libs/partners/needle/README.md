---
layout: integration
name: Needle Langchain
description: Build RAG Langchain pipelines with Needle
authors:
  - name: needle-ai.com
pypi: https://pypi.org/project/needle-python/
repo: https://github.com/JANHMS/langchain-needle
sdk: https://github.com/oeken/needle-python
sdk-pypi: https://pypi.org/project/needle-python/
type: RAG as a Service
report_issue: https://github.com/Needle-ai/needle-python/issues
logo: /logos/needle-logo.png
version: Langchain 0.3
toc: true
---

# Needle Langchain Integration

### **Table of Contents**

- [Installation](#installation)
- [Usage](#usage)

## Installation

This project resides in the Python Package Index (PyPI), so it can easily be installed with `pip`:

```console
pip install langchain-needle
```

## Usage

The Needle Langchain integration provides components for working with the Needle API. These components can be used to create collections, add files to collections, and perform searches.

### Configure your API keys

- Get your `NEEDLE_API_KEY` from [Developer settings](https://needle-ai.com/dashboard/settings/developer).

```
os.environ["NEEDLE_API_KEY"] = ""
```

- Provide your `OPENAI_API_KEY`

```
os.environ["OPENAI_API_KEY"] = ""
```

### Create a Collection in Needle and add Files

```python
from langchain_needle import NeedleLoader

# NeedleLoader Initialization
document_store = NeedleLoader(
    needle_api_key=os.getenv("NEEDLE_API_KEY"),
    collection_name="Langchain",
)

files = {
    "tech-radar-30.pdf": "https://www.thoughtworks.com/content/dam/thoughtworks/documents/radar/2024/04/tr_technology_radar_vol_30_en.pdf"
}

document_store.add_files(files=files)
```

### Search your collection

```python
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_needle import NeedleRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

# Initialize the Needle retriever (make sure your Needle API key is set as an environment variable)
retriever = NeedleRetriever(
    needle_api_key=os.getenv("NEEDLE_API_KEY"),
    collection_id="COLLECTION_ID"
)

# Define system prompt for the assistant
system_prompt = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know, say so concisely.\n\n{context}
    """

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Define the question-answering chain using a document chain (stuff chain) and the retriever
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create the RAG (Retrieval-Augmented Generation) chain by combining the retriever and the question-answering chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Define the input query
query = { "input": "Did RAG move to accepted?" }

response = rag_chain.invoke(query)

response
```
