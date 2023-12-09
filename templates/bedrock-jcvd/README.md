# Bedrock JCVD 🕺🥋

## Overview

LangChain template that uses [Anthropic's Claude on Amazon Bedrock](https://aws.amazon.com/bedrock/claude/) to behave like JCVD.

> I am the Fred Astaire of Chatbots! 🕺

'![JCVD](https://media.tenor.com/CVp9l7g3axwAAAAj/jean-claude-van-damme-jcvd.gif)

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package bedrock-jcvd
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add bedrock-jcvd
```

And add the following code to your `server.py` file:
```python
from bedrock_jcvd import chain as bedrock_jcvd_chain

add_routes(app, bedrock_jcvd_chain, path="/bedrock-jcvd")
```

(Optional) Let's now configure LangSmith. 
LangSmith will help us trace, monitor and debug LangChain applications. 
LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/). 
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

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

We can also access the playground at [http://127.0.0.1:8000/bedrock-jcvd/playground](http://127.0.0.1:8000/bedrock-jcvd/playground)

![JCVD Playground](jcvd_langserve.png)