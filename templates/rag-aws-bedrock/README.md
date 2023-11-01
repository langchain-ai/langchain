
# rag-aws-bedrock

This template is designed to connect with the AWS Bedrock service, a managed server that offers a set of foundation models.

It primarily uses the `Anthropic Claude` for text generation and `Amazon Titan` for text embedding, and utilizes FAISS as the vectorstore.

For additional context on the RAG pipeline, refer to [this notebook](https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/03_QuestionAnswering/01_qa_w_rag_claude.ipynb).

## Environment Setup

Before you can use this package, ensure that you have configured `boto3` to work with your AWS account. 

For details on how to set up and configure `boto3`, visit [this page](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration).

In addition, you need to install the `faiss-cpu` package to work with the FAISS vector store:

```bash
pip install faiss-cpu
```

You should also set the following environment variables to reflect your AWS profile and region (if you're not using the `default` AWS profile and `us-east-1` region):

* `AWS_DEFAULT_REGION`
* `AWS_PROFILE`

## Usage

First, install the LangChain CLI:

```shell
pip install -U "langchain-cli[serve]"
```

To create a new LangChain project and install this as the only package:

```shell
langchain app new my-app --package rag-aws-bedrock
```

To add this package to an existing project:

```shell
langchain app add rag-aws-bedrock
```

Then add the following code to your `server.py` file:
```python
from rag_aws_bedrock import chain as rag_aws_bedrock_chain

add_routes(app, rag_aws_bedrock_chain, path="/rag-aws-bedrock")
```

(Optional) If you have access to LangSmith, you can configure it to trace, monitor, and debug LangChain applications. If you don't have access, you can skip this section.

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

If you are inside this directory, you can spin up a LangServe instance directly by:

```shell
langchain serve
```

This will start the FastAPI app with a server running locally at [http://localhost:8000](http://localhost:8000)

You can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and access the playground at [http://127.0.0.1:8000/rag-aws-bedrock/playground](http://127.0.0.1:8000/rag-aws-bedrock/playground).  

You can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-aws-bedrock")
```