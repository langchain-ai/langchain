import os

from openai import error

from langchain.embeddings import OpenAIEmbeddings


def test_azure_openai_embeddings():
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_BASE"] = "https://<your-endpoint.openai.azure.com/"
    os.environ["OPENAI_API_KEY"] = "your AzureOpenAI key"
    os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

    embeddings = OpenAIEmbeddings(deployment="your-embeddings-deployment-name")
    text = "This is a test document."

    try:
        query_result = embeddings.embed_query(text)
    except error.InvalidRequestError as e:
        if "Must provide an 'engine' or 'deployment_id' parameter" in str(e):
            assert False, "deployment was provided to OpenAIEmbeddings by openai.Embeddings didn't get it."
    except Exception as e:
        # Expected to fail because endpoint doesn't exist.
        pass