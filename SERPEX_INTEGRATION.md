# LangChain Serpex Integration

Official Serpex search integration for LangChain Python.

## Installation

```bash
pip install langchain-serpex
```

## Setup

Get your API key from [Serpex](https://serpex.dev) and set it as an environment variable:

```bash
export SERPEX_API_KEY="your-api-key-here"
```

## Usage

### Basic Example

```python
from langchain_serpex import SerpexSearchResults

# Using environment variable
tool = SerpexSearchResults()

result = tool.invoke("latest AI developments")
print(result)
```

### With Custom Parameters

```python
from langchain_serpex import SerpexSearchResults

tool = SerpexSearchResults(
    api_key="your-api-key",    # Optional if SERPEX_API_KEY is set
    engine="google",            # auto, google, bing, duckduckgo, brave, yahoo, yandex
    category="web",             # currently only "web" supported
    time_range="day"            # all, day, week, month, year (not supported by Brave)
)

result = tool.invoke("coffee shops near me")
print(result)
```

### With LangChain Agent

```python
from langchain_serpex import SerpexSearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# Initialize the search tool
search = SerpexSearchResults(
    api_key="your-api-key",
    engine="auto",
    time_range="week"
)

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create agent
agent = initialize_agent(
    tools=[search],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
result = agent.run("What are the latest developments in AI?")
print(result)
```

### Async Usage

```python
import asyncio
from langchain_serpex import SerpexSearchResults

async def search_async():
    tool = SerpexSearchResults(api_key="your-api-key")
    result = await tool.ainvoke("Python programming best practices")
    print(result)

asyncio.run(search_async())
```

## API Parameters

### Supported Parameters

- **`api_key`** (optional): Your Serpex API key
  - If not provided, reads from `SERPEX_API_KEY` environment variable

- **`engine`** (optional): Search engine to use
  - Options: `"auto"`, `"google"`, `"bing"`, `"duckduckgo"`, `"brave"`, `"yahoo"`, `"yandex"`
  - Default: `"auto"` (automatically routes with retry logic)

- **`category`** (optional): Search category
  - Options: `"web"` (more categories coming soon)
  - Default: `"web"`

- **`time_range`** (optional): Filter results by time
  - Options: `"all"`, `"day"`, `"week"`, `"month"`, `"year"`
  - Note: Not supported by Brave engine

- **`base_url`** (optional): Custom API endpoint
  - Default: `"https://api.serpex.com"`
  - Can also be set via `SERPEX_BASE_URL` environment variable

### Response Format

The tool returns formatted search results as a string containing:

1. **Instant Answers**: Direct answers from knowledge panels
2. **Infoboxes**: Knowledge panel descriptions
3. **Organic Results**: Web search results with titles, URLs, and snippets
4. **Suggestions**: Related search queries
5. **Corrections**: Suggested query corrections

Example response:
```
Found 10 results:

[1] Starbucks - Coffee Shop
URL: https://www.starbucks.com/store-locator/store/1234
Premium coffee, perfect lattes, and great atmosphere in the heart of downtown...

[2] Local Coffee Roasters
URL: https://localcoffee.com
Artisanal coffee beans, locally sourced and expertly roasted daily...
```

## Advanced Usage

### Custom Query Parameters

```python
from langchain_serpex import SerpexSearchResults

tool = SerpexSearchResults(
    api_key="your-api-key",
    engine="bing",
    time_range="month"
)

# Override parameters for specific search
result = tool.invoke(
    "AI news",
    engine="google",  # Override default engine
    time_range="day"   # Override default time range
)
```

### Error Handling

```python
from langchain_serpex import SerpexSearchResults

try:
    tool = SerpexSearchResults(api_key="your-api-key")
    result = tool.invoke("search query")
    print(result)
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Search error: {e}")
```

## Features

- **Multi-Engine Support**: Automatically routes requests across multiple search engines
- **Smart Retry Logic**: Built-in retry mechanism for failed requests
- **Real-Time Results**: Get fresh search results from the web
- **Async Support**: Full async/await support for concurrent operations
- **Simple Integration**: Easy to use with LangChain agents and chains
- **Structured Output**: Clean, formatted search results ready for LLM consumption
- **Type Safety**: Full Pydantic validation and type hints

## Cost

All search engines cost 1 credit per successful request. Credits never expire.

## Rate Limits

- 300 requests per second
- No daily limits

## Requirements

- Python 3.8+
- httpx
- langchain-core
- pydantic

## Documentation

For detailed API documentation, visit: [https://serpex.dev/docs](https://serpex.dev/docs)

## Support

- Email: support@serpex.dev
- Documentation: https://serpex.dev/docs
- Dashboard: https://serpex.dev/dashboard

## License

MIT

## Development

To run tests:

```bash
pytest tests/
```

To run with coverage:

```bash
pytest --cov=langchain_serpex tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
