# langchain-serpex

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-serpex?label=%20)](https://pypi.org/project/langchain-serpex/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-serpex)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)

This package contains the LangChain integration with SERPEX.

## Installation

```bash
pip install langchain-serpex
```

## What is SERPEX?

SERPEX is a powerful multi-engine search API that provides access to search results from Google, Bing, DuckDuckGo, Baidu, Yandex, and other search engines in JSON format. It's designed for developers building AI applications, SEO tools, market research platforms, and data aggregation services.

## Features

- **Multi-Engine Support**: Search across Google, Bing, DuckDuckGo, Baidu, and Yandex
- **Rich Results**: Get organic results, answer boxes, knowledge graphs, news, images, videos, and shopping results
- **Localization**: Support for location-based and language-specific searches
- **Real-time Data**: Access to current search results
- **Easy Integration**: Simple API with comprehensive documentation

## Quick Start

### Get Your API Key

Sign up at [SERPEX](https://serpex.io) to get your API key.

### Basic Usage

```python
from langchain_serpex import SerpexSearchResults

# Initialize the tool
tool = SerpexSearchResults(
    api_key="your-serpex-api-key",
    engine="google",
    num_results=10
)

# Perform a search
results = tool.invoke("latest AI developments")
print(results)
```

### With Agents

```python
from langchain_serpex import SerpexSearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# Initialize the search tool
search_tool = SerpexSearchResults(
    api_key="your-serpex-api-key",
    engine="google",
    num_results=5
)

# Initialize the LLM
llm = ChatOpenAI(temperature=0)

# Create an agent with the search tool
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
result = agent.run("What are the latest developments in quantum computing?")
print(result)
```

### Advanced Configuration

```python
from langchain_serpex import SerpexSearchResults

# Configure with advanced parameters
tool = SerpexSearchResults(
    api_key="your-serpex-api-key",
    engine="google",
    num_results=20,
    gl="us",  # Country code
    hl="en",  # Language code
    location="New York",  # Specific location
    time_period="m",  # Results from past month
    safe_search="moderate"
)

# Search for location-specific results
results = tool.invoke("best restaurants")
print(results)
```

### Different Search Engines

```python
from langchain_serpex import SerpexSearchResults

# Google Search
google_tool = SerpexSearchResults(api_key="your-key", engine="google")
google_results = google_tool.invoke("Python programming")

# Bing Search
bing_tool = SerpexSearchResults(api_key="your-key", engine="bing")
bing_results = bing_tool.invoke("Python programming")

# DuckDuckGo Search
ddg_tool = SerpexSearchResults(api_key="your-key", engine="duckduckgo")
ddg_results = ddg_tool.invoke("Python programming")
```

## Configuration

### Environment Variables

You can set your SERPEX API key as an environment variable:

```bash
export SERPEX_API_KEY="your-serpex-api-key"
```

Then use the tool without passing the API key:

```python
from langchain_serpex import SerpexSearchResults

tool = SerpexSearchResults()  # Will use SERPEX_API_KEY from environment
```

### Parameters

- `api_key` (str): Your SERPEX API key (required)
- `engine` (str): Search engine to use - "google", "bing", "duckduckgo", "baidu", "yandex" (default: "google")
- `num_results` (int): Number of results to return, 1-100 (default: 10)
- `gl` (str): Country code for localized results (e.g., "us", "uk", "ca")
- `hl` (str): Language code (e.g., "en", "es", "fr")
- `location` (str): Specific location for localized results
- `time_period` (str): Time filter - "d" (day), "w" (week), "m" (month), "y" (year)
- `safe_search` (str): Safe search filter - "off", "moderate", "strict"

## Documentation

For more detailed documentation, visit:
- [LangChain Documentation](https://python.langchain.com)
- [SERPEX API Documentation](https://serpex.io/docs)

## Support

For issues and questions:
- GitHub Issues: [langchain-serpex issues](https://github.com/langchain-ai/langchain/issues)
- SERPEX Support: [support@serpex.io](mailto:support@serpex.io)

## License

This package is licensed under the MIT License.
