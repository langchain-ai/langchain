# LangServe Templates

LangServe Templates are the easiest and fastest way to build a production ready LLM application.
These templates serve as a set of reference architectures for a wide variety of popular LLM use cases.
They can be easily downloaded into a LangServe project, which makes it easy to deploy them in a production ready manner.

Below we cover how to get started. Some other helpful docs:

- [Index of Templates](INDEX.md)
- [Contributing](CONTRIBUTING.md)

## Usage

To use, first install the LangChain CLI.

```shell
pip install -U "langchain-cli[serve]"
```

Next, create a new LangChain project:

```shell
langchain serve new my-app
```

This will create a new directory called `my-app` with two folders:

- `app`: This is where LangServe code will live
- `packages`: This is where your chains or agents will live

To pull in an existing template as a package, you first need to go into your new project:

```shell
cd my-app
```

And you can the add a template as a project.
In this getting started guide, we will add a simple `pirate-speak` project.
All this project does is convert user input into pirate speak.

```shell
langchain serve add pirate-speak
```

This will pull in the specified template into `packages/pirate-speak`

You then need to install this package so you can use it in the langserve app:

```shell
pip install -e packages/pirate-speak
```

We install it with `-e` so that if you modify the template at all (which you likely will) the changes are updated.

In order to have LangServe use this project, you then need to modify `app/server.py`.
Specifically, you should add something like:

```python
from fastapi import FastAPI
from langserve import add_routes
# This depends on the structure of the package you install
from pirate_speak import chain

app = FastAPI()

add_routes(app, chain, path="pirate-speak")
```

You can then spin up production-ready endpoints, along with a playground, by running:

```shell
langchain start
```

This now gives a fully deployed LangServe application.
For example, you get a playground out-of-the-box at [http://127.0.0.1:8000/pirate-speak/playground/](http://127.0.0.1:8000/pirate-speak/playground/):

You also get API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)


You can also use the LangServe SDK to easily interact with the API endpoint as if it was another [Runnable](https://python.langchain.com/docs/expression_language/) object

```python
from langserve import RemoteRunnable

api = RemoteRunnable("http://127.0.0.1:8000/pirate-speak")
api.invoke({"text": "hi"})
```


## Other Resources

- [Index of Templates](INDEX.md)
- [Contributing](CONTRIBUTING.md)
