from typing import Any, Dict, List

from langchain.pydantic_v1 import root_validator
from langchain.schema.embeddings import Embeddings
from langchain_community.utilities.vertexai import raise_vertex_import_error
from langchain_community.llms.vertexai import _VertexAICommon


class VertexAIMultimodalEmbeddings(_VertexAICommon, Embeddings):
    model: Any
    model_name: str = "multimodalembedding@001"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates that the python package exists in environment."""
        cls._try_init_vertexai(values)
        try:
            from vertexai.preview.vision_models import MultiModalEmbeddingModel
            values["model"] = MultiModalEmbeddingModel.from_pretrained(values['model_name'])
        except ImportError:
            raise_vertex_import_error()
        return values

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        try:
            from vertexai.preview.vision_models import Image
            from PIL import Image as _PILImage
        except ImportError:
            raise ImportError("Please install the PIL library: pip install pillow")

        image_features = []
        for uri in uris:
            img = Image.load_from_file(uri)
            embedding = self.model.get_embeddings(image=img)
            image_features.append(embedding.image_embedding)

        return image_features

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        text_features = []
        for text in texts:
            embedding = self.model.get_embeddings(contextual_text=text)
            text_features.append(embedding)

        return text_features

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

