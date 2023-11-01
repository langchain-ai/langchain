from typing import Any, Dict, List

from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.schema.embeddings import Embeddings


class OpenCLIPEmbeddings(BaseModel, Embeddings):
    """OpenClip embedding models.

    To use, you should have the open_clip python package installed

    Example:
        .. code-block:: python

            from langchain.embeddings import OpenCLIPEmbeddings

            embeddings = OpenCLIPEmbeddings()
    """

    client: Any  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that open_clip library is installed."""

        try:
            import open_clip
            model_name = 'ViT-B-32'
            checkpoint = 'laion2b_s34b_b79k'
            model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, 
                                                                         pretrained=checkpoint)
            values["model"] = model
            values["preprocess"] = preprocess
        except ImportError:
            raise ImportError(
                "Could not import open_clip library. "
                "Please install the open_clip library to "
                "use this embedding model: pip install open_clip_torch"
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using open_clip.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "Could not import torch library. "
                "Please install the torch library to "
                "use this embedding model: pip install torch"
            )
        with torch.no_grad():
            text_features = [self.model.encode_text(text) for text in texts]
            text_features = [text_feature / text_feature.norm(dim=-1, keepdim=True) for text_feature in text_features]
            return [list(map(float, e)) for e in text_features]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using open_clip.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

    def embed_image(self, images: List[float]) -> List[List[float]]:
        """Embed a list of images using open_clip.

        Args:
            images: The list of images (as numpy arrays) to embed.

        Returns:
            List of embeddings, one for each image.
        """
        try:
            from PIL import Image as _PILImage
        except ImportError:
            raise ImportError(
                "Could not import PIL library. "
                "Please install the PIL library to "
                "use this embedding model: pip install pillow"
            )
        try:
            import torch
        except ImportError:
            raise ImportError(
                "Could not import torch library. "
                "Please install the torch library to "
                "use this embedding model: pip install torch"
            )
        
        pil_images = [_PILImage.fromarray(image) for image in images]
         
        with torch.no_grad():
            image_features = [self.model.encode_image(self.preprocess(pil_image).unsqueeze(0)) for pil_image in pil_images]
            image_features = [image_feature / image_feature.norm(dim=-1, keepdim=True) for image_feature in image_features]
            return [list(map(float, e)) for e in image_features]