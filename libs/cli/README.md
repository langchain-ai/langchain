# langchain-cli

Install CLI

`pip install -U --pre langchain-cli`

Create new langchain app

`langchain serve new my-app`

Go into app

`cd my-app`

Install a package

`langchain serve add extraction-summary`

Install langserve

`pip install "langserve[all]"`

Install the langchain package

`pip install -e packages/extraction-summary`

Edit `app/server.py` to add that package to the routes

```markdown
from fastapi import FastAPI
from langserve import add_routes 
from extraction_summary.chain import chain

app = FastAPI()

add_routes(app, chain)
```

Set env vars

```shell
export OPENAI_API_KEY=...
```
```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY="<your-api-key>"
export LANGCHAIN_PROJECT="extraction-summary"
```

Run the app

`python app/server.py`
