# OpenAI Functions Agent - Gmail

Ever struggled to reach inbox zero? 

Using this template, you can create and customize your very own AI assistant to manage your Gmail account. Using the default Gmail tools, it can read, search through, and draft emails to respond on your behalf. It also has access to a Tavily search engine so it can search for relevant information about any topics or people in the email thread before writing, ensuring the drafts include all the relevant information needed to sound well-informed.

![Animated GIF showing the interface of the Gmail Agent Playground with a cursor interacting with the input field.](./static/gmail-agent-playground.gif "Gmail Agent Playground Interface")

## The details

This assistant uses OpenAI's [function calling](https://python.langchain.com/docs/modules/chains/how_to/openai_functions) support to reliably select and invoke the tools you've provided

This template also imports directly from [langchain-core](https://pypi.org/project/langchain-core/) and [`langchain-community`](https://pypi.org/project/langchain-community/) where appropriate. We have restructured LangChain to let you select the specific integrations needed for your use case. While you can still import from `langchain` (we are making this transition backwards-compatible), we have separated the homes of most of the classes to reflect ownership and to make your dependency lists lighter. Most of the integrations you need can be found in the `langchain-community` package, and if you are just using the core expression language API's, you can even build solely based on `langchain-core`.

## Environment Setup

The following environment variables need to be set:

Set the `OPENAI_API_KEY` environment variable to access the OpenAI models.

Set the `TAVILY_API_KEY` environment variable to access Tavily search.

Create a [`credentials.json`](https://developers.google.com/gmail/api/quickstart/python#authorize_credentials_for_a_desktop_application) file containing your OAuth client ID from Gmail. To customize authentication, see the [Customize Auth](#customize-auth) section below.

_*Note:* The first time you run this app, it will force you to go through a user authentication flow._

(Optional): Set `GMAIL_AGENT_ENABLE_SEND`  to `true` (or modify the `agent.py` file in this template) to give it access to the "Send" tool. This will give your assistant permissions to send emails on your behalf without your explicit review, which is not recommended.


## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package openai-functions-agent-gmail
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add openai-functions-agent-gmail
```

And add the following code to your `server.py` file:
```python
from openai_functions_agent import agent_executor as openai_functions_agent_chain

add_routes(app, openai_functions_agent_chain, path="/openai-functions-agent-gmail")
```

(Optional) Let's now configure LangSmith. 
LangSmith will help us trace, monitor and debug LangChain applications. 
You can sign up for LangSmith [here](https://smith.langchain.com/). 
If you don't have access, you can skip this section

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

If you are inside this directory, then you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server is running locally at 
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/openai-functions-agent-gmail/playground](http://127.0.0.1:8000/openai-functions-agent/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/openai-functions-agent-gmail")
```

## Customize Auth

```
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials

# Can review scopes here https://developers.google.com/gmail/api/auth/scopes
# For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)
```