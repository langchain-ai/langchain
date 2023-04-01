import ai21
import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Union, Any, Optional, Dict
from langchain.embeddings.base import Embeddings

class AI21Embeddings(Embeddings):
    def __init__(self, api_key: str):
        """
        Initialize AI21Embeddings with the provided API key.
        """
        ai21.api_key = api_key

    def __enter__(self) -> 'AI21Embeddings':
        """
        Enable usage of the 'with' statement for this class.
        """
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        """
        Close the AI21Embeddings instance when used with the 'with' statement.
        """
        pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.generate_embeddings(texts)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.generate_embeddings([text])
        return embeddings[0].tolist()

    def generate_embeddings(self, texts: List[str], model: str = "j2-grande-instruct") -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts using the specified model.

        :param texts: A list of texts for which to generate embeddings.
        :param model: The name of the AI21 model to use for generating embeddings.
        :return: A list of numpy arrays containing the embeddings.
        """
        prompt = "\n".join([f"Embed the following text as a 768-dimensional vector: {text}" for text in texts])

        try:
            response: dict = ai21.Completion.execute(
                model=model,
                prompt=prompt,
                numResults=1,
                maxTokens=768 * len(texts),
                temperature=0,
                topKReturn=0,
                topP=1
            )
            tokens = response["completions"][0]["data"]["tokens"]
            embeddings = [np.array([token["generatedToken"]["logprob"] for token in tokens[i*768:(i+1)*768]]) for i in range(len(texts))]
            return embeddings
        except Exception as e:
            print(f"Error while generating embeddings: {e}")
            return []

    def get_similarity(self, text1: str, text2: str, model: str = "j2-grande-instruct") -> Union[float, None]:
        """
        Calculate the similarity between two texts using the specified model.

        :param text1: The first text.
        :param text2: The second text.
        :param model: The name of the AI21 model to use for generating embeddings.
        :return: A float representing the similarity between the two texts.
        """
        emb1, emb2 = self.generate_embeddings([text1, text2], model=model)
        if not emb1 or not emb2:
            return None

        try:
            similarity = 1 - cosine(emb1, emb2)
            return similarity
        except Exception as e:
            print(f"Error while calculating similarity: {e}")
            return None
