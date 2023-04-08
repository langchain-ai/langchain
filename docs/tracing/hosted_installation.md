# Cloud Hosted Setup

We offer a hosted version of tracing at [langchainplus.vercel.app](https://langchainplus.vercel.app/). You can use this to view traces from your run without having to run the server locally.

Note: we are currently only offering this to a limited number of users. The hosted platform is VERY alpha, in active development, and data might be dropped at any time. Don't depend on data being persisted in the system long term and don't log traces that may contain sensitive information. If you're interested in using the hosted platform, please fill out the form [here](https://forms.gle/tRCEMSeopZf6TE3b6).

## Installation

1. Login to the system and click "API Key" in the top right corner. Generate a new key and keep it safe. You will need it to authenticate with the system.

## Environment Setup

After installation, you must now set up your environment to use tracing.

This can be done by setting an environment variable in your terminal by running `export LANGCHAIN_HANDLER=langchain`.

You can also do this by adding the below snippet to the top of every script. **IMPORTANT:** this must go at the VERY TOP of your script, before you import anything from `langchain`. 

```python
import os
os.environ["LANGCHAIN_HANDLER"] = "langchain"
```

You will also need to set an environment variable to specify the endpoint and your API key. This can be done with the following environment variables:

1. `LANGCHAIN_ENDPOINT` = "https://langchain-api-gateway-57eoxz8z.uc.gateway.dev"
2. `LANGCHAIN_API_KEY` - set this to the API key you generated during installation.

An example of adding all relevant environment variables is below:

```python
import os
os.environ["LANGCHAIN_HANDLER"] = "langchain"
os.environ["LANGCHAIN_ENDPOINT"] = "https://langchain-api-gateway-57eoxz8z.uc.gateway.dev"
os.environ["LANGCHAIN_API_KEY"] = "my_api_key"  # Don't commit this to your repo! Better to set it in your terminal.
```
