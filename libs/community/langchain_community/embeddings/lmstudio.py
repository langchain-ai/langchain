import requests
from langchain_core.embeddings import Embeddings


class LMStudioEmbeddings(Embeddings):
    # These are default parameters as example, you can send your own when instantiating the class
    # Example: lm_studio_embeddings = LMStudioEmbeddings(base_url=custom_base_url, model_identifier=custom_model_identifier)
    base_url: str = "http://localhost:1234/v1"
    model_identifier: str = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.f16.gguf"

    def embed_documents(self, documents):
        url = f"{self.base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        data = {
            "input": documents,
            "model": self.model_identifier
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        # Need to convert to a list of embeddings instead of list of objects of embeddings, to support the way
        # LangChain's vector databases expect output from embedding models
        # Extract the embeddings
        embeddings = response.json()["data"]
        extracted_embeddings = [item['embedding'] for item in embeddings]
        return extracted_embeddings
        # return response.json()["data"]

    def embed_query(self, query):
        return self.embed_documents([query])[0]