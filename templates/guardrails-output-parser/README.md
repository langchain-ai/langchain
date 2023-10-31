
# guardrails-output-parser

This template uses [guardrails-ai](https://github.com/guardrails-ai/guardrails) to validate LLM output. 

The `GuardrailsOutputParser` is set in `chain.py`.
 
The default example protects against profanity.

## Environment Setup

Set the `OPENAI_API_KEY` environment variable to access the OpenAI models.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package guardrails-output-parser
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add guardrails-output-parser
```

And add the following code to your `server.py` file:
```python
from guardrails_output_parser import chain as guardrails_output_parser_chain

add_routes(app, guardrails_output_parser_chain, path="/guardrails-output-parser")
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

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/guardrails-output-parser/playground](http://127.0.0.1:8000/guardrails-output-parser/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/guardrails-output-parser")
```

If Guardrails does not find any profanity, then the translated output is returned as is. If Guardrails does find profanity, then an empty string is returned.
