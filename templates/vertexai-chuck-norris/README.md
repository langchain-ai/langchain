
# vertexai-chuck-norris

This template makes jokes about Chuck Norris using Vertex AI PaLM2. 

## Environment Setup

First, make sure you have a Google Cloud project with
an active billing account, and have the [gcloud CLI installed](https://cloud.google.com/sdk/docs/install).

Configure [application default credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc):

```shell
gcloud auth application-default login
```

To set a default Google Cloud project to use, run this command and set [the project ID](https://support.google.com/googleapi/answer/7014113?hl=en) of the project you want to use:
```shell
gcloud config set project [PROJECT-ID]
```

Enable the [Vertex AI API](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com) for the project:
```shell
gcloud services enable aiplatform.googleapis.com
```

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package pirate-speak
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add vertexai-chuck-norris
```

And add the following code to your `server.py` file:
```python
from vertexai_chuck_norris.chain import chain as vertexai_chuck_norris_chain

add_routes(app, vertexai_chuck_norris_chain, path="/vertexai-chuck-norris")
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
We can access the playground at [http://127.0.0.1:8000/vertexai-chuck-norris/playground](http://127.0.0.1:8000/vertexai-chuck-norris/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/vertexai-chuck-norris")
```
