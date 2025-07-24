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

### Advanced Features

You can use advanced features like text limits, summaries, and live crawling:

```python
from langchain_exa import ExaSearchRetriever, TextContentsOptions

# Create a new instance with advanced options
exa = ExaSearchRetriever(
    exa_api_key="YOUR API KEY",
    k=20,  # Number of results (1-100)
    type="auto",  # Can be "neural", "keyword", or "auto"
    livecrawl="always",  # Can be "always", "fallback", or "never"
    summary=True,  # Get an AI-generated summary of each result
    text_contents_options={"max_characters": 3000}  # Limit text length
)

# Search for a query with custom summary prompt
exa_with_custom_summary = ExaSearchRetriever(
    exa_api_key="YOUR API KEY",
    summary={"query": "generate one line summary in simple words."}  # Custom summary prompt
)
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

## Configuration Options

All Exa tools support the following common parameters:

- `num_results` (1-100): Number of search results to return
- `type`: Search type - "neural", "keyword", or "auto"
- `livecrawl`: Live crawling mode - "always", "fallback", or "never"
- `summary`: Get AI-generated summaries (True/False or custom prompt dict)
- `text_contents_options`: Dict to limit text length (e.g. `{"max_characters": 2000}`)
- `highlights`: Include highlighted text snippets (True/False)
