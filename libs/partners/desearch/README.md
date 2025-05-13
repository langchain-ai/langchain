# LangChain Desearch Integration

This project integrates the Desearch API with LangChain tools to enable various search and data-fetching functionalities, such as web searches, Twitter data retrieval, and AI-powered searches.

## Features

- **Grouped Tools**:
  - **Search Tools**: General-purpose search tools for AI, web, and Twitter searches.
  - **Twitter Tools**: Tools specifically for Twitter-related operations.

## Installation

Install the package using pip:

```bash
pip install -U langchain-desearch

```

## Usage

### Grouped Tools

#### Search Tools
The `search_tools` group contains tools for general-purpose searches:
- `DesearchTool`: Perform AI searches, web link searches, and Twitter post searches.
- `BasicWebSearchTool`: Conduct basic web searches.
- `BasicTwitterSearchTool`: Perform advanced Twitter searches with filters.


### Examples

#### Using Tools
```python
from langchain_desearch.tools import DesearchTool, BasicWebSearchTool, BasicTwitterSearchTool
from dotenv import load_dotenv
load_dotenv()

# Example 1: Using DesearchTool
tool = DesearchTool()
result = tool._run(
    prompt="Bittensor",
    tool=['web'],
    model="NOVA",
    date_filter="PAST_24_HOURS",
    streaming=False
)
print(result)

# Example 2: Using BasicWebSearchTool
tool = BasicWebSearchTool()
result = tool._run(
    query="Latest news on AI",
    num=5,
    start=1
)
print(result)

# Example 3: Using BasicTwitterSearchTool
tool = BasicTwitterSearchTool()
result = tool._run(
    query="AI trends",
    sort="Top",
    count=5
)
print(result)
```
