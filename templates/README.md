# LangServe Templates

Templates for a fully functioning app that can be hosted by LangServe.

## Usage

To use, first install the LangChain CLI.

```shell
pip install -U langchain-cli
```

Then, install `langserve`:

```shell
pip install "langserve[all]"
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

And you can the add a template as a project

```shell
langchain serve add $PROJECT_NAME
```

This will pull in the specified template into `packages/$PROJECT_NAME`

You then need to install this package so you can use it in the langserve app:

```shell
pip install -e packages/$PROJECT_NAME
```

We install it with `-e` so that if we modify the template at all (which we likely will) the changes are updated.

In order to have LangServe use this project, you then need to modify `app/server.py`.
Specifically, you should add something like:

```python
from fastapi import FastAPI
from langserve import add_routes
# This depends on the structure of the package you install
from my_project import chain

app = FastAPI()

add_routes(app, chain)
```

You can then spin up production-ready endpoints, along with a playground, by running:

```shell
python app/server.py
```

## Adding a template

See [here](CONTRIBUTING.md)
