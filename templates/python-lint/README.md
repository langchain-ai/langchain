# python-lint

This agent specializes in generating high-quality Python code with a focus on proper formatting and linting. It uses `black`, `ruff`, and `mypy` to ensure the code meets standard quality checks.

This streamlines the coding process by integrating and responding to these checks, resulting in reliable and consistent code output.

It cannot actually execute the code it writes, as code execution may introduce additional dependencies and potential security vulnerabilities.
This makes the agent both a secure and efficient solution for code generation tasks.

You can use it to generate Python code directly, or network it with planning and execution agents.

## Environment Setup

- Install `black`, `ruff`, and `mypy`: `pip install -U black ruff mypy`
- Set `OPENAI_API_KEY` environment variable.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package python-lint
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add python-lint
```

And add the following code to your `server.py` file:
```python
from python_lint import agent_executor as python_lint_agent

add_routes(app, python_lint_agent, path="/python-lint")
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
We can access the playground at [http://127.0.0.1:8000/python-lint/playground](http://127.0.0.1:8000/python-lint/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/python-lint")
```
