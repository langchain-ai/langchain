# RAG AWS Bedrock

AWS Bedrock is a managed serve that offers a set of foundation models.

Here we will use `Anthropic Claude` for text generation and `Amazon Titan` for text embedding.

We will use FAISS as our vectorstore.

(See [this notebook](https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/03_QuestionAnswering/01_qa_w_rag_claude.ipynb) for additional context on the RAG pipeline.)

Code here uses the `boto3` library to connect with the Bedrock service. See [this page](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration) for setting up and configuring boto3 to work with an AWS account. 

##  FAISS

You need to install the `faiss-cpu` package to work with the FAISS vector store.

```bash
pip install faiss-cpu
```


##  LLM and Embeddings

The code assumes that you are working with the `default` AWS profile and `us-east-1` region. If not, specify these environment variables to reflect the correct region and AWS profile.

* `AWS_DEFAULT_REGION`
* `AWS_PROFILE`
