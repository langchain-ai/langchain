# SSearX Web Search Tool

The **SSearX Web Search Tool** is a custom search tool designed for the LangChain framework, allowing users to perform automated Google searches and scrape specific websites for content. This tool can retrieve relevant links, access target websites, and extract content, making it highly useful for tasks that require external web data.

## Features

- **Google Search Integration**: Automates searches on Google and retrieves relevant search results.
- **Website Scraping**: Accesses specific websites and scrapes content directly from the page.
- **Content Summarization**: Provides detailed information from search results or websites based on user queries.
- **Compliance with Legal Restrictions**: Avoids searches or content extraction related to topics prohibited by Brazilian law.

## Installation

To use the SSearX Web Search Tool in LangChain, clone the repository and make sure you have the following dependencies installed:

```bash
pip install beautifulsoup4 langchain-community
```
## Directory Structure
Place the SSearX tool in the following directory structure within the LangChain repository:

```bash
langchain/
└── tools/
    └── ssx_web/
        ├── ssx_web.py               # Main tool file
        └── README.md                 # Documentation for the tool
```
## Usage

The SSearX class provides two primary methods:

 - ** ssxGoogle(query: str) - Performs a Google search with the given query and returns content from the search results. **
 - ** ssxWebsite(url: str) - Accesses a specified website and retrieves its content. **

### Example:
Google Search
Use ssxGoogle to perform a Google search for a specific term:

```python
from langchain.tools.ssx_web.ssx_web import SSearX

ssx = SSearX()
search_results = ssx.ssxGoogle("LangChain")
print(search_results)
```

## Website Scraping
Use ssxWebsite to scrape content from a specific website:
```python
website_content = ssx.ssxWebsite("example.com")
print(website_content)

```
## API Reference
Class: SSearX
Method: ssxGoogle(query: str) -> str

Description: Performs a Google search and retrieves content from the search results page.
Args: query - A string containing the search term.
Returns: The content from the Google search results.
Method: ssxWebsite(url: str) -> str

Description: Accesses a specified website and scrapes the page content.
Args: url - The base URL of the website (without https://www.).
Returns: The scraped content from the target website.
Environment Variables
Ensure that you have the USER_AGENT environment variable set to a valid user-agent string. This is essential for sending requests to websites that require a recognizable user-agent.


## Environment Variables
Ensure that you have the USER_AGENT environment variable set to a valid user-agent string. This is essential for sending requests to websites that require a recognizable user-agent.

```bash
export USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
```

On Windows, you can set it in PowerShell with:
```powershell
$env:USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
```

## Running Tests
Tests are included to verify the functionality of SSearX. You can find the tests in tests/tools/test_ssx_web.py. To run the tests, use the following command:

```python
python -m unittest discover tests
```

## Contributing
If you'd like to contribute to the SSearX Web Search Tool, please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.
```yaml

---

This 'README.md' file is ready to be added to the `ssx_web` folder in your LangChain project. It provides clear and organized documentation for other developers to understand and use the 'SSearX' tool.

```

## AUTHOR
```yaml

---
DEV Master: Thiago Bluhm thiagobluhm@gmail.com

