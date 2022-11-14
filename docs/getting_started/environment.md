# Setting up your environment

Using LangChain will usually require integrations with one or more model providers, data stores, apis, etc.
There are two components to setting this up, installing the correct python packages and setting the right environment variables.

## Python packages
The python package needed varies based on the integration. See the list of integrations for details.
There should also be helpful error messages raised if you try to run an integration and are missing any required python packages.

## Environment Variables
The environment variable needed varies based on the integration. See the list of integrations for details.
There should also be helpful error messages raised if you try to run an integration and are missing any required environment variables.

You can set the environment variable in a few ways. 
If you are trying to set the environment variable `FOO` to value `bar`, here are the ways you could do so:
- From the command line:
```
export FOO=bar
```
- From the python notebook/script:
```python
import os
os.environ["FOO"] = "bar"
```

For the Getting Started example, we will be using OpenAI's APIs, so we will first need to install their SDK:

```
pip install openai
```

We will then need to set the environment variable. Let's do this from inside the Jupyter notebook (or Python script).

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```
