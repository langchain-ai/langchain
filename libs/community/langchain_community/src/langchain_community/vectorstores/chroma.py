# chroma.py
from langchain_core.documents import Document

class Chroma:
    def __init__(self, collection_name=None, embedding_function=None):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.texts = []
        self.ids = []

    def add_texts(self, texts, ids=None):
        self.texts = texts
        self.ids = ids if ids else [f"{self.collection_name}_doc_{i}" for i in range(len(texts))]

    def similarity_search(self, query, k=1):
        # Return top-k dummy Document objects
        return [
            Document(page_content=self.texts[i], metadata={"id": self.ids[i]})
            for i in range(min(k, len(self.texts)))
        ]
