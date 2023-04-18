"""
Loader that loads image captions
By default, the loader utilizes the pre-trained BLIP image captioning model.
https://huggingface.co/Salesforce/blip-image-captioning-base

"""
from typing import Any, List, Tuple, Union

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class ImageCaptionLoader(BaseLoader):
    """Loader that loads the captions of an image"""

    def __init__(
        self,
        path_images: Union[str, List[str]],
        blip_processor: str = "Salesforce/blip-image-captioning-base",
        blip_model: str = "Salesforce/blip-image-captioning-base",
    ):
        """
        Initialize with a list of image paths
        """
        if isinstance(path_images, str):
            self.image_paths = [path_images]
        else:
            self.image_paths = path_images

        self.blip_processor = blip_processor
        self.blip_model = blip_model

    def load(self) -> List[Document]:
        """
        Load from a list of image files
        """
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError:
            raise ValueError(
                "transformers package not found, please install with"
                "`pip install transformers`"
            )

        processor = BlipProcessor.from_pretrained(self.blip_processor)
        model = BlipForConditionalGeneration.from_pretrained(self.blip_model)

        results = []
        for path_image in self.image_paths:
            caption, metadata = self._get_captions_and_metadata(
                model=model, processor=processor, path_image=path_image
            )
            doc = Document(page_content=caption, metadata=metadata)
            results.append(doc)

        return results

    def _get_captions_and_metadata(
        self, model: Any, processor: Any, path_image: str
    ) -> Tuple[str, dict]:
        """
        Helper function for getting the captions and metadata of an image
        """
        try:
            from PIL import Image
        except ImportError:
            raise ValueError(
                "PIL package not found, please install with `pip install pillow`"
            )

        try:
            if path_image.startswith("http://") or path_image.startswith("https://"):
                image = Image.open(requests.get(path_image, stream=True).raw).convert(
                    "RGB"
                )
            else:
                image = Image.open(path_image).convert("RGB")
        except Exception:
            raise ValueError(f"Could not get image data for {path_image}")

        inputs = processor(image, "an image of", return_tensors="pt")
        output = model.generate(**inputs)

        caption: str = processor.decode(output[0])
        metadata: dict = {"image_path": path_image}

        return caption, metadata
