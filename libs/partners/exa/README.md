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

## Exa Search Results

You can run the ExaSearchResults module as follows

```python
from langchain_exa import ExaSearchResults

# Initialize the ExaSearchResults tool
search_tool = ExaSearchResults(exa_api_key="03ab2ac5-66d5-46c1-8f30-91585712609a")

# Perform a search query
search_results = search_tool._run(
    query="When was the last time the New York Knicks won the NBA Championship?",
    num_results=5,
    text_contents_options=True,
    highlights=True
)

print("Search Results:", search_results)
```