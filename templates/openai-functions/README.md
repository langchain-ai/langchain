# Function calling with OpenAI

This template enables OpenAI [function calling](https://python.langchain.com/docs/modules/chains/how_to/openai_functions).

Function calling can be used for various tasks, such as extraction or tagging. 

Specify the function you want to use in `chain.py`

By default, it will tag the input text using the following fields:

* summarize
* provide keywords
* provide language

##  LLM

This template will use `OpenAI` by default. 

Be sure that `OPENAI_API_KEY` is set in your enviorment.

## Adding the template

Install the langchain package
```
pip install -e packages/openai_functions
```

Edit app/server.py to add that package to the routes
```
from fastapi import FastAPI
from langserve import add_routes 
from openai_functions.chain import chain

app = FastAPI()
add_routes(app, chain)
```

Run the app
```
python app/server.py
```

You can use this template in the Playground:

http://127.0.0.1:8000/openai-functions/playground/

Also, see Jupyter notebook `openai_functions` for various other ways to connect to the template.