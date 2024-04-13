
# rag-timescale-conversation

This template is used for [conversational](https://python.langchain.com/docs/expression_language/cookbook/retrieval#conversational-retrieval-chain) [retrieval](https://python.langchain.com/docs/use_cases/question_answering/), which is one of the most popular LLM use-cases.

It passes both a conversation history and retrieved documents into an LLM for synthesis.

## Environment Setup

This template uses Timescale Vector as a vectorstore and requires that `TIMESCALES_SERVICE_URL`. Signup for a 90-day trial [here](https://console.cloud.timescale.com/signup?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral) if you don't yet have an account.

To load the sample dataset, set `LOAD_SAMPLE_DATA=1`. To load your own dataset see the section below.

Set the `OPENAI_API_KEY` environment variable to access the OpenAI models.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-timescale-conversation
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-timescale-conversation
```

And add the following code to your `server.py` file:
```python
from rag_timescale_conversation import chain as rag_timescale_conversation_chain

add_routes(app, rag_timescale_conversation_chain, path="/rag-timescale_conversation")
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
We can access the playground at [http://127.0.0.1:8000/rag-timescale-conversation/playground](http://127.0.0.1:8000/rag-timescale-conversation/playground)

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-timescale-conversation")
```

See the `rag_conversation.ipynb` notebook for example usage.

## Loading your own dataset

To load your own dataset you will have to create a `load_dataset` function. You can see an example, in the
`load_ts_git_dataset` function defined in the `load_sample_dataset.py` file. You can then run this as a
standalone function (e.g. in a bash script) or add it to chain.py (but then you should run it just once).