
# rag-gpt-crawler

GPT-crawler will crawl websites to produce files for use in custom GPTs or other apps (RAG).

This template uses [gpt-crawler](https://github.com/BuilderIO/gpt-crawler) to build a RAG app

## Environment Setup

Set the `OPENAI_API_KEY` environment variable to access the OpenAI models.

## Crawling

Run GPT-crawler to extact content from a set of urls, using the config file in GPT-crawler repo.

Here is example config for LangChain use-case docs:

```
export const config: Config = {
  url: "https://python.langchain.com/docs/use_cases/",
  match: "https://python.langchain.com/docs/use_cases/**",
  selector: ".docMainContainer_gTbr",
  maxPagesToCrawl: 10,
  outputFileName: "output.json",
};
```

Then, run this as described in the [gpt-crawler](https://github.com/BuilderIO/gpt-crawler) README:

```
npm start
```

And copy the `output.json` file into the folder containing this README.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-gpt-crawler
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-gpt-crawler
```

And add the following code to your `server.py` file:
```python
from rag_chroma import chain as rag_gpt_crawler

add_routes(app, rag_gpt_crawler, path="/rag-gpt-crawler")
```

(Optional) Let's now configure LangSmith. 
LangSmith will help us trace, monitor and debug LangChain applications. 
You can sign up for LangSmith [here](https://smith.langchain.com/). 
If you don't have access, you can skip this section

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server is running locally at 
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/rag-gpt-crawler/playground](http://127.0.0.1:8000/rag-gpt-crawler/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-gpt-crawler")
```