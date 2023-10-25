# Extraction

This template performs an extraction task on the input text using the Pydantic schema provided.

It will use OpenAI function calling to produce output that adheres to the schema.

##  LLM

Be sure that `OPENAI_API_KEY` is set in order to the OpenAI model(s).

## Installation

Get langserve and create app:
```
pip install < to add > 
langchain serve new my-app
cd my-app
```

Add template and start server:
```
langchain serve add extraction-summary
```

Start server:
```
langchain serve install
poetry run poe start
```

Note, we can now look at the endpoints:

http://127.0.0.1:8000/docs#

And look specifically at our loaded template:

http://127.0.0.1:8000/docs#/default/invoke_extraction_summary_invoke_post
 
We can also use remote runnable to call it (see `text_extraction.ipynb`):
```
from langserve.client import RemoteRunnable
extraction_model = RemoteRunnable('http://localhost:8000/extraction-summary')
extraction_model.invoke(< text >)
```
