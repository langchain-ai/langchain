from langchain.document_loaders.unstructured import UnstructuredTextLoader
import requests


class WebpageLoader(UnstructuredTextLoader):

    def __init__(self, webpage: str):
        r = requests.get(webpage)
        metadata = {"source": webpage}
        super().__init__(text=r.text, metadata=metadata)
