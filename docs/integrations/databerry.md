# Databerry

This page covers how to use the [Databerry](https://databerry.ai) within LangChain.

## What is Databerry?

Databerry is an [open source](https://github.com/gmpetrov/databerry) document retrievial platform that helps to connect your personal data with Large Language Models.

![Databerry](../_static/DataberryDashboard.png)

## Quick start

Retrieving documents stored in Databerry from LangChain is very easy!

```python
from langchain.retrievers import DataberryRetriever

retriever = DataberryRetriever(
    datastore_url="https://api.databerry.ai/query/clg1xg2h80000l708dymr0fxc",
    # api_key="DATABERRY_API_KEY", # optional if datastore is public
    # top_k=10 # optional
)

docs = retriever.get_relevant_documents("What's Databerry?")
```
