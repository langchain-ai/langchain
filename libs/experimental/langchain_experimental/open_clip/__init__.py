"""**OpenCLIP Embeddings** model.

OpenCLIP is a multimodal model that can encode text and images into a shared space.

See this paper for more details: https://arxiv.org/abs/2103.00020
and [this repository](https://github.com/mlfoundations/open_clip) for details.

"""
from .open_clip import OpenCLIPEmbeddings

__all__ = ["OpenCLIPEmbeddings"]
