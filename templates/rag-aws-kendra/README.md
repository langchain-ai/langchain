# RAG AWS Kendra

[Amazon Kendra](https://aws.amazon.com/kendra/) is an intelligent search service powered by machine learning (ML).
Here we will use `Anthropic Claude` for text generation and `Amazon Kendra` for retrieving documents. Together, with these two services, this application uses a Retrieval chain to answer questions from your documents.

(See [this page](https://aws.amazon.com/blogs/machine-learning/quickly-build-high-accuracy-generative-ai-applications-on-enterprise-data-using-amazon-kendra-langchain-and-large-language-models/) for additional context on building RAG applications with Amazon Kendra.)

Code here uses the `boto3` library to connect with the Bedrock service. See [this page](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration) for setting up and configuring boto3 to work with an AWS account. 

## Kendra Index

You will need a Kendra Index setup before using this template. For setting up a sample index, you can use this [Cloudformation template](https://github.com/aws-samples/amazon-kendra-langchain-extensions/blob/main/kendra_retriever_samples/kendra-docs-index.yaml) to create the index. This template includes sample data containing AWS online documentation for Amazon Kendra, Amazon Lex, and Amazon SageMaker. Alternately, if you have an Amazon Kendra index and have indexed your own dataset, you can use that. Launching the stack requires about 30 minutes followed by about 15 minutes to synchronize it and ingest the data in the index. Therefore, wait for about 45 minutes after launching the stack. Note the Index ID and AWS Region on the stack’s Outputs tab.

##  Environment variables

The code assumes that you are working with the `default` AWS profile and `us-east-1` region. If not, specify these environment variables to reflect the correct region and AWS profile. 

* `AWS_DEFAULT_REGION`
* `AWS_PROFILE`

This code also requires specifying the `KENDRA_INDEX_ID` env variable which should have the Index ID of the Kendra index. Note that the Index ID is a 36 character alphanumeric value that can be found in the index detail page.
