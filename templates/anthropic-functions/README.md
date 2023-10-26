# Function calling with Anthropic

This template enables [Anthropic function calling](https://python.langchain.com/docs/integrations/chat/anthropic_functions).

Function calling can be used for various tasks, such as extraction or tagging. 

Specify the function you want to use in `chain.py`

By default, it will tag the input text using the following fields:

* sentiment
* aggressiveness
* language

##  LLM

This template will use `Claude2` by default. 

Be sure that `ANTHROPIC_API_KEY` is set in your enviorment.

## Adding the template

Create your LangServe app:
```
langchain serve new my-app
cd my-app
```

Add template:
```
langchain serve add anthropic-functions
```

Start server:
```
langchain start
```

See Jupyter notebook `anthropic_functions` for various way to connect to the template.