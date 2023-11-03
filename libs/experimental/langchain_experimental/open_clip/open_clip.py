from typing import Any, Dict, List

import numpy as np

from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.schema.embeddings import Embeddings


class OpenCLIPEmbeddings(BaseModel, Embeddings):
    model: Any
    preprocess: Any
    tokenizer: Any

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that open_clip and torch libraries are installed."""
        try:
            import open_clip

            model_name = "ViT-B-32"
            checkpoint = "laion2b_s34b_b79k"
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=model_name, pretrained=checkpoint
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            values["model"] = model
            values["preprocess"] = preprocess
            values["tokenizer"] = tokenizer

        except ImportError:
            raise ImportError(
                "Please ensure both open_clip and torch libraries are installed. "
                "pip install open_clip_torch torch"
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        text_features = [
            self.model.encode_text(self.tokenizer(text)).tolist() for text in texts
        ]
        return text_features

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_image(self, images: List[np.ndarray]) -> List[List[float]]:
        try:
            from PIL import Image as _PILImage
        except ImportError:
            raise ImportError("Please install the PIL library: pip install pillow")
        pil_images = [_PILImage.fromarray(image) for image in images]
        image_features = [
            self.model.encode_image(self.preprocess(pil_image).unsqueeze(0)).tolist()
            for pil_image in pil_images
        ]
        return image_features
