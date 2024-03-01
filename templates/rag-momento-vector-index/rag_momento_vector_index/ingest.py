### Ingest code - you may need to run this the first time
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import MomentoVectorIndex
from langchain_text_splitters import RecursiveCharacterTextSplitter
from momento import (
    CredentialProvider,
    PreviewVectorIndexClient,
    VectorIndexConfigurations,
)


def load(API_KEY_ENV_VAR_NAME: str, index_name: str) -> None:
    if os.environ.get(API_KEY_ENV_VAR_NAME, None) is None:
        raise Exception(f"Missing `{API_KEY_ENV_VAR_NAME}` environment variable.")

    # Load
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    data = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    # Add to vectorDB
    MomentoVectorIndex.from_documents(
        all_splits,
        embedding=OpenAIEmbeddings(),
        client=PreviewVectorIndexClient(
            configuration=VectorIndexConfigurations.Default.latest(),
            credential_provider=CredentialProvider.from_environment_variable(
                API_KEY_ENV_VAR_NAME
            ),
        ),
        index_name=index_name,
    )
