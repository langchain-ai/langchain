# Cloud Hosted Tracing Setup

This guide provides instructions for setting up your environment to use the cloud-hosted version of the LangChain Plus tracing server. For instructions on locally hosted tracing, please reference the [Locally Hosted Tracing Setup](./local_installation.md) guide.

We offer a hosted version of tracing at the [LangChain Plus website](https://www.langchain.plus/). You can use this to interact with your traces and evaluation datasets without having to install the local server.

**Note**: We are currently only offering this to a limited number of users. The hosted platform is in the alpha stage, actively under development, and data might be dropped at any time. Do not depend on data being persisted in the system long term and refrain from logging traces that may contain sensitive information. If you're interested in using the hosted platform, please fill out the form [here](https://forms.gle/tRCEMSeopZf6TE3b6).

## Setup

Follow these steps to set up your environment to use the cloud-hosted tracing server:

1. Log in to the system and click "API Key" in the top right corner. Generate a new key and assign it to the `LANGCHAIN_API_KEY` environment variable.

```bash
export LANGCHAIN_API_KEY="your api key"
```

## Environment Configuration

Once you've set up your account, configure your LangChain application's environment to use tracing. This can be done by setting an environment variable in your terminal by running:

```bash
export LANGCHAIN_TRACING_V2=true
```

You can also add the following snippet to the top of every script:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
```

Additionally, you need to set an environment variables to specify the endpoint. You can do this with the following environment variable:

```bash
export LANGCHAIN_ENDPOINT="https://api.langchain.plus"
```

Here's an example of adding all relevant environment variables:

```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_ENDPOINT="https://api.langchain.plus"
export LANGCHAIN_API_KEY="my api key"
# export LANGCHAIN_SESSION="My Session Name" # Optional, otherwise, traces are logged to the "default" session 
```

Or in python:
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://langchain-api-gateway-57eoxz8z.uc.gateway.dev"
os.environ["LANGCHAIN_API_KEY"] = "my_api_key"  # Don't commit this to your repo! Set it in your terminal instead.
# os.environ["LANGCHAIN_SESSION"] = "My Session Name" # Optional, otherwise, traces are logged to the "default" session 
```

## Tracing Context Manager

Although using environment variables is recommended for most tracing use cases, you can also configure runs to be sent to a specific session using the context manager:

```python
from langchain.callbacks.manager import tracing_v2_enabled

with tracing_v2_enabled("My Session Name"):
    ...
```


## Navigating the LangChainPlus UI

You can check out an overview of the LangChainPlus UI in the [LangChain Tracing](../additional_resources/tracing.md) guide.