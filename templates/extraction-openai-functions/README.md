# Extraction with OpenAI Function Calling

This template shows how to do extraction of structured data from unstructured data, using OpenAI [function calling](https://python.langchain.com/docs/modules/chains/how_to/openai_functions).

Specify the information you want to extract in `chain.py`

By default, it will extract the title and author of papers.

##  LLM

This template will use `OpenAI` by default. 

Be sure that `OPENAI_API_KEY` is set in your environment.

## Adding the template

Install the langchain package
```
pip install -e packages/extraction_openai_functions
```

Edit app/server.py to add that package to the routes
```
from fastapi import FastAPI
from langserve import add_routes 
from extraction_openai_functions.chain import chain

app = FastAPI()
add_routes(app, chain)
```

Run the app
```
python app/server.py
```

You can use this template in the Playground:

http://127.0.0.1:8000/extraction-openai-functions/playground/

Also, see Jupyter notebook `openai_functions` for various other ways to connect to the template.