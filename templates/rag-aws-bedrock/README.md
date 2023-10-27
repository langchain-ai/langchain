# RAG AWS Bedrock

AWS Bedrock is a managed serve that offers a set of foundation models.

Here we will use `Anthropic Claude` for text generation and `Amazon Titan` for text embedding.

We will use Pinecode as our vectorstore.

(See [this notebook](https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/03_QuestionAnswering/01_qa_w_rag_claude.ipynb) for additional context on the RAG pipeline.)

(See [this notebook](https://github.com/aws-samples/amazon-bedrock-workshop/blob/58f238a183e7e629c9ae11dd970393af4e64ec44/00_Intro/bedrock_boto3_setup.ipynb#Prerequisites) for additional context on setup.)

##  Pinecone

This connects to a hosted Pinecone vectorstore.

Be sure that you have set a few env variables in `chain.py`:

* `PINECONE_API_KEY`
* `PINECONE_ENV`
* `index_name`

##  LLM and Embeddings

Be sure to set AWS enviorment variables:

* `AWS_DEFAULT_REGION`
* `AWS_PROFILE`
* `BEDROCK_ASSUME_ROLE`
