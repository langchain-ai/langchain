from typing import Any, Dict, List

from langchain.pydantic_v1 import BaseModel, root_validator
from langchain_core.embeddings import Embeddings


class OpenCLIPEmbeddings(BaseModel, Embeddings):
    """OpenCLIP Embeddings model."""

    model: Any
    preprocess: Any
    tokenizer: Any
    # Select model: https://github.com/mlfoundations/open_clip
    model_name: str = "ViT-H-14"
    checkpoint: str = "laion2b_s32b_b79k"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that open_clip and torch libraries are installed."""
        try:
            import open_clip

            # Fall back to class defaults if not provided
            model_name = values.get("model_name", cls.__fields__["model_name"].default)
            checkpoint = values.get("checkpoint", cls.__fields__["checkpoint"].default)

            # Load model
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
        text_features = []
        for text in texts:
            # Tokenize the text
            tokenized_text = self.tokenizer(text)

            # Encode the text to get the embeddings
            embeddings_tensor = self.model.encode_text(tokenized_text)

            # Normalize the embeddings
            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)

            # Convert normalized tensor to list and add to the text_features list
            embeddings_list = normalized_embeddings_tensor.squeeze(0).tolist()
            text_features.append(embeddings_list)

        return text_features

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        try:
            from PIL import Image as _PILImage
        except ImportError:
            raise ImportError("Please install the PIL library: pip install pillow")

        # Open images directly as PIL images
        pil_images = [_PILImage.open(uri) for uri in uris]

        image_features = []
        for pil_image in pil_images:
            # Preprocess the image for the model
            preprocessed_image = self.preprocess(pil_image).unsqueeze(0)

            # Encode the image to get the embeddings
            embeddings_tensor = self.model.encode_image(preprocessed_image)

            # Normalize the embeddings tensor
            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)

            # Convert tensor to list and add to the image_features list
            embeddings_list = normalized_embeddings_tensor.squeeze(0).tolist()

            image_features.append(embeddings_list)

        return image_features
