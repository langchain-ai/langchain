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
results  = exa.invoke("What is the capital of France?")

# Print the results
print(results)
```

## Exa Search Results

You can run the ExaSearchResults module as follows

```python
from langchain_exa import ExaSearchResults

# Initialize the ExaSearchResults tool
search_tool = ExaSearchResults(exa_api_key="YOUR API KEY")

# Perform a search query
search_results = search_tool._run(
    query="When was the last time the New York Knicks won the NBA Championship?",
    num_results=5,
    text_contents_options=True,
    highlights=True
)

print("Search Results:", search_results)
```

## Exa Find Similar Results

You can run the ExaFindSimilarResults module as follows

```python
from langchain_exa import ExaFindSimilarResults

# Initialize the ExaFindSimilarResults tool
find_similar_tool = ExaFindSimilarResults(exa_api_key="YOUR API KEY")

# Find similar results based on a URL
similar_results = find_similar_tool._run(
    url="http://espn.com",
    num_results=5,
    text_contents_options=True,
    highlights=True
)

print("Similar Results:", similar_results)
```