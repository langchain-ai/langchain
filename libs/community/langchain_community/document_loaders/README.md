# Steel Web Loader

This loader uses [Steel.dev](https://steel.dev) to load web pages with support for proxy networks and automated CAPTCHA solving. Steel provides managed browser infrastructure that makes it easy to automate web interactions reliably.

## Installation

```bash
pip install steel-browser-python
```

## Usage

Basic usage:

```python
from langchain_community.document_loaders import SteelWebLoader

# Initialize the loader
loader = SteelWebLoader(
    "https://example.com",
    steel_api_key="your-api-key"  # Or set STEEL_API_KEY environment variable
)

# Load the page
documents = loader.load()
```

### Advanced Configuration

The loader supports several configuration options:

```python
loader = SteelWebLoader(
    "https://example.com",
    steel_api_key="your-api-key",
    extract_strategy="html",  # 'text', 'markdown', or 'html'
    timeout=60000,           # Navigation timeout in milliseconds
    use_proxy=True,         # Use Steel's proxy network
    solve_captcha=True      # Enable automated CAPTCHA solving
)
```

### Environment Variables

- `STEEL_API_KEY`: Your Steel API key (required if not passed to constructor)

## Features

- **Proxy Network**: Access to Steel's global proxy network
- **CAPTCHA Solving**: Automated CAPTCHA solving capabilities
- **Multiple Extract Strategies**: Extract content as text, markdown, or HTML
- **Session Management**: Automatic browser session lifecycle management
- **Live Session Viewer**: Debug sessions with Steel's session viewer

## Example

Here's a complete example showing how to use the loader with an agent:

```python
from langchain_community.document_loaders import SteelWebLoader
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

# Initialize the loader
loader = SteelWebLoader("https://example.com")

# Create a tool that uses the loader
tools = [
    Tool(
        name="SteelWebLoader",
        func=lambda url: SteelWebLoader(url).load(),
        description="Load a webpage using Steel browser automation"
    )
]

# Initialize the agent
llm = OpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

# Use the agent
agent.run("What is the main content of example.com?")
