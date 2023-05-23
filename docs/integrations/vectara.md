# Vectara


What is Vectara?

**Vectara Overview:**
- Vectara is developer-first API platform for building conversational search applications
- To use Vectara - first [sign up](https://console.vectara.com/signup) and create an account. Then create a corpus and an API key for indexing and searching.
- You can use Vectara's [indexing API](https://docs.vectara.com/docs/indexing-apis/indexing) to add documents into Vectara's index
- You can use Vectara's [Search API](https://docs.vectara.com/docs/search-apis/search) to query Vectara's index (which also supports Hybrid search implicitly).
- You can use Vectara's integration with LangChain as a Vector store or using the Retriever abstraction.

## Installation and Setup
To use Vectara with LangChain no special installation steps are required. You just have to provide your customer_id, corpus ID, and an API key created within the Vectara console to enable indexing and searching.

### VectorStore

There exists a wrapper around the Vectara platform, allowing you to use it as a vectorstore, whether for semantic search or example selection.

To import this vectorstore:
```python
from langchain.vectorstores import Vectara
```

To create an instance of the Vectara vectorstore:
```python
vectara = Vectara(customer_id, corpus_id, api_key)
```
the customer_id, corpus_id and api_key are optioonal, and if they are not supplied will be read from the environment variables `VECTARA_CUSTOMER_ID`, `VECTARA_CORPUS_ID` and `VECTARA_API_KEY`, respectivelu


For a more detailed walkthrough of the Vectara wrapper, see one of the two example notebooks:
* [Chat Over Documents with Vectara](./vectara/vectara_chat.html)
* [Vectara Text Generation](./vectara/vectara_text_generation.html)


