import getpass
import os

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.milvus import Milvus
from langchain_nvidia_aiplay import NVIDIAEmbeddings
from langchain_text_splitters.character import CharacterTextSplitter

if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")  # noqa: T201
else:
    nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key

# Note: if you change this, you should also change it in `nvidia_rag_canonical/chain.py`
EMBEDDING_MODEL = "nvolveqa_40k"
HOST = "127.0.0.1"
PORT = "19530"
COLLECTION_NAME = "test"

embeddings = NVIDIAEmbeddings(model=EMBEDDING_MODEL)

if __name__ == "__main__":
    # Load docs
    loader = PyPDFLoader("https://www.ssa.gov/news/press/factsheets/basicfact-alt.pdf")
    data = loader.load()

    # Split docs
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    docs = text_splitter.split_documents(data)

    # Insert the documents in Milvus Vector Store
    vector_db = Milvus.from_documents(
        docs,
        embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": HOST, "port": PORT},
    )
