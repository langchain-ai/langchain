# Environment setup

Using LangChain will usually require integrations with one or more model providers, data stores, apis, etc.
There are two components to setting this up, installing the correct python packages and setting the right environment variables.

## Python packages
The python package needed varies based on the integration. See the list of integrations for details.
There should also be helpful error messages raised if you try to run an integration and are missing any required python packages.

## Environment Variables
The environment variable needed varies based on the integration. See the list of integrations for details.
There should also be helpful error messages raised if you try to run an integration and are missing any required environment variables.

You can set the environment variable in a few ways, either from the command line or from the python notebook/script.

For the Getting Started example, we will be using OpenAI's APIs, so we will first need to install their SDK:

```
pip install openai
```

We will then need to set the environment variable. 
If we do this from inside the Jupyter notebook (or Python script), this is how we would do that:

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```

Alternatively, we could also do this from the command line before starting our Jupyter notebook.
In that case, it would look like:

```python
export FOO=bar
```
