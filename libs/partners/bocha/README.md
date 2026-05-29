# langchain-bocha

LangChain integration for [Bocha AI](https://bocha.ai).

## Installation

```bash
pip install langchain-bocha
```

## Usage

```python
from langchain_bocha import ChatBocha, BochaSearchRun

# Use Chat Model
llm = ChatBocha(model="deepseek-v4-pro")
response = llm.invoke("Hello!")
print(response.content)

# Use Web Search Tool
search = BochaSearchRun()
results = search.run("Beijing weather")
print(results)
```

## Environment Variables

Set your Bocha API key:

```bash
export BOCHA_API_KEY="your-api-key-here"
```
