# Locally Hosted Tracing Setup

This guide provides instructions for installing and setting up your environment to use the locally hosted version of the LangChain Plus tracing server. For instructions on a hosted tracing solution, please reference the [Hosted Tracing Setup](./hosted_installation.md) guide.

## Installation

1. Ensure Docker is installed and running on your system. To install Docker, refer to the [Get Docker](https://docs.docker.com/get-docker/) documentation.
2. Install the latest version of `langchain` by running the following command:
   ```bash
   pip install -U langchain
   ```
3. Start the LangChain Plus tracing server by executing the following command in your terminal:
   ```bash
   langchain plus start
   ```
   _Note: The `langchain` command was installed when you installed the LangChain library using (`pip install langchain`)._

4. After the server has started, it will open the [Local UI](http://localhost). In the terminal, it will also display environment variables that you can configure to send your traces to the server. For more details on this, refer to the Environment Setup section below.

5. To stop the server, run the following command in your terminal:
   ```bash
   langchain plus stop
   ```

## Environment Configuration

With the LangChain Plus tracing server running, you can begin sending traces by setting the `LANGCHAIN_TRACING_V2` environment variable:

```bash
export LANGCHAIN_TRACING_V2=true
```

Or at the top of every python script:
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
```


Here's an example of adding all relevant environment variables:

```bash
export LANGCHAIN_TRACING_V2="true"
# export LANGCHAIN_SESSION="My Session Name" # Optional, otherwise, traces are logged to the "default" session 
```

Or in python:
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_SESSION"] = "My Session Name" # Optional, otherwise, traces are logged to the "default" session 
```

## Tracing Context Manager

Although using environment variables is recommended for most tracing use cases, you can configure runs to be sent to a specific session using the context manager:

```python
from langchain.callbacks.manager import tracing_v2_enabled

with tracing_v2_enabled("My Session Name"):
    ...
```

## Connecting from a Remote Server

To connect to LangChainPlus when running applications on a remote server, such as a [Google Colab notebook](https://colab.research.google.com/) or a [HuggingFace Space](https://huggingface.co/docs/hub/spaces), we offer two simple options:

1. Use our [hosted tracing](./hosted_installation.md) server.
2. Expose a public URL to your local tracing service.

Below are the full instructions to expose start a local LangChainPlus server and connect from a remote server:

1. Ensure Docker is installed and running on your system. To install Docker, refer to the [Get Docker](https://docs.docker.com/get-docker/) documentation.
2. Install the latest version of `langchain` by running the following command:
   ```bash
   pip install -U langchain
   ```
3. Start the LangChain Plus tracing server and expose  by executing the following command in your terminal:
   ```bash
   langchain plus start --expose
   ```
   Note: The `--expose` flag is required to expose your local server to the internet. By default, ngrok permits tunneling for up to 2 hours at a time. For longer sessions, you can make an [ngrok account](https://ngrok.com/) and use your auth token:

   ```bash
   langchain plus start --expose --ngrok-authtoken "your auth token"
   ```
   
4. After the server has started, it will open the [Local LangChainPlus UI](http://localhost) a well as the [ngrok dashboard](http://0.0.0.0:4040/inspect/http). In the terminal, it will also display environment variables needed to send traces to the server via the tunnel URL. These will look something like the following:

   ```bash
      LANGCHAIN_TRACING_V2=true
      LANGCHAIN_ENDPOINT=https://1234-01-23-45-678.ngrok.io
   ```

5. In your remote LangChain application, set the environment variables using the output from your terminal in the previous step:

   ```python
   import os
   os.environ["LANGCHAIN_TRACING_V2"] = True
   os.environ["LANGCHAIN_ENDPOINT"] = "https://1234-01-23-45-678.ngrok.io" # Replace with your ngrok tunnel URL
   ```

6. Run your LangChain code and visualize the traces in the [LangChainPlus UI](http://localhost/sessions)

7. To stop the server, run the following command in your terminal:
   ```bash
   langchain plus stop
   ```

## Navigating the LangChainPlus UI

You can check out an overview of the LangChainPlus UI in the [LangChain Tracing](../additional_resources/tracing.md) guide.