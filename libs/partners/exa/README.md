# langchain-exa

This package contains the LangChain integrations for Exa Cloud generative models.

## Installation

```bash
pip install -U langchain-exa
```

## Exa Search Retriever

You can retrieve search results as follows

```python
from langchain_exa import ExaSearchRetriever

exa_api_key = "YOUR API KEY"

# Create a new instance of the ExaSearchRetriever
exa = ExaSearchRetriever(exa_api_key=exa_api_key)

# Search for a query and save the results
results  = exa.get_relevant_documents(query="What is the capital of France?")

# Print the results
print(results)
```