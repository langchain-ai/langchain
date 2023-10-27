# Extraction with OpenAI Function Calling

This template shows how to do extraction of structured data from unstructured data, using OpenAI [function calling](https://python.langchain.com/docs/modules/chains/how_to/openai_functions).

Specify the information you want to extract in `chain.py`

By default, it will extract the title and author of papers.

##  LLM

This template will use `OpenAI` by default. 

Be sure that `OPENAI_API_KEY` is set in your environment.

## App

Example `server.py`:
```python
from fastapi import FastAPI
from langserve import add_routes
from extraction_openai_functions.chain import chain

app = FastAPI()

# Edit this to add the chain you want to add
add_routes(app, chain, path="extraction_openai_functions")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
```

Run:
```shell
langchain start
```

Check endpoint:
```shell
http://0.0.0.0:8001/docs
```

See `extraction_openai_functions.ipynb` for example usage - 
```python
from langserve.client import RemoteRunnable
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
text = loader.load()

oai_function = RemoteRunnable('http://0.0.0.0:8001/extraction_openai_functions')
oai_function.invoke({"input":text[0].page_content[0:4000]})
```
