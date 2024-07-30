# rag-aws-kendra

This template is an application that utilizes Amazon Kendra, a machine learning powered search service, and Anthropic Claude for text generation. The application retrieves documents using a Retrieval chain to answer questions from your documents. 

It uses the `boto3` library to connect with the Bedrock service. 

For more context on building RAG applications with Amazon Kendra, check [this page](https://aws.amazon.com/blogs/machine-learning/quickly-build-high-accuracy-generative-ai-applications-on-enterprise-data-using-amazon-kendra-langchain-and-large-language-models/).

## Environment Setup

Please ensure to setup and configure `boto3` to work with your AWS account. 

You can follow the guide [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration).

You should also have a Kendra Index set up before using this template. 

You can use [this Cloudformation template](https://github.com/aws-samples/amazon-kendra-langchain-extensions/blob/main/kendra_retriever_samples/kendra-docs-index.yaml) to create a sample index. 

This includes sample data containing AWS online documentation for Amazon Kendra, Amazon Lex, and Amazon SageMaker. Alternatively, you can use your own Amazon Kendra index if you have indexed your own dataset. 

The following environment variables need to be set:

* `AWS_DEFAULT_REGION` - This should reflect the correct AWS region. Default is `us-east-1`.
* `AWS_PROFILE` - This should reflect your AWS profile. Default is `default`.
* `KENDRA_INDEX_ID` - This should have the Index ID of the Kendra index. Note that the Index ID is a 36 character alphanumeric value that can be found in the index detail page.

## Usage

To use this package, you should first have the LangChain CLI installed:

```shell
pip install -U langchain-cli
```

To create a new LangChain project and install this as the only package, you can do:

```shell
langchain app new my-app --package rag-aws-kendra
```

If you want to add this to an existing project, you can just run:

```shell
langchain app add rag-aws-kendra
```

And add the following code to your `server.py` file:
```python
from rag_aws_kendra.chain import chain as rag_aws_kendra_chain

add_routes(app, rag_aws_kendra_chain, path="/rag-aws-kendra")
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

This will start the FastAPI app with a server running locally at 
[http://localhost:8000](http://localhost:8000)

We can see all templates at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
We can access the playground at [http://127.0.0.1:8000/rag-aws-kendra/playground](http://127.0.0.1:8000/rag-aws-kendra/playground)  

We can access the template from code with:

```python
from langserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/rag-aws-kendra")
```
