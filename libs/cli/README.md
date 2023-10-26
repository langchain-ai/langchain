# langchain-cli

Install CLI

`pip install -U --pre langchain-cli`

Create new langchain app

`langchain serve new my-app`

Go into app

`cd my-app`

Install a package

`langchain serve add extraction-summary`

(Activate virtualenv)

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

Run the app

`python app/server.py`
