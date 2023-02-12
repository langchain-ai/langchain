"""Wrapper around HuggingFace embedding models to perform inference on a self-hosted remote hardware."""
from typing import Any, List

from pydantic import BaseModel, Extra

from langchain.embeddings.base import Embeddings

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_INSTRUCT_MODEL = "hkunlp/instructor-large"
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)


def _embed_documents(
    model_id: str, instruct: bool = False, *args: Any, **kwargs: Any
) -> List[List[float]]:
    """Inference function to send to the remote hardware. Accepts a sentence_transformer model_id and
    returns a list of embeddings for each document in the batch.
    """
    if not instruct:
        import sentence_transformers

        client = sentence_transformers.SentenceTransformer(model_id)
    else:
        from InstructorEmbedding import INSTRUCTOR

        client = INSTRUCTOR(model_id)
    import torch

    if torch.cuda.is_available():
        client = client.cuda()
    return client.encode(*args, **kwargs)


class SelfHostedHuggingFaceEmbeddings(BaseModel, Embeddings):
    """Wrapper around sentence_transformers embedding models to perform inference on self-hosted remote hardware.
    Supported hardware includes auto-launched instances on AWS, GCP, Azure, and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another cloud like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import SelfHostedHFEmbeddings
            import runhouse as rh
            model_name = "sentence-transformers/all-mpnet-base-v2"
            gpu = rh.cluster(name="rh-a10x", instance_type="A100:1")
            hf = SelfHostedHFEmbeddings(model_name=model_name, hardware=gpu)
    """

    client: Any  #: :meta private:
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    model_reqs: List[str] = ["sentence_transformers", "torch"]
    """Requirements to install on hardware to inference the model."""
    hardware: Any
    """Remote hardware to send the inference function to."""

    def __init__(self, **kwargs: Any):
        """Initialize the remote inference function."""
        super().__init__(**kwargs)
        try:
            import runhouse as rh

            self.client = rh.send(fn=_embed_documents).to(
                self.hardware, reqs=["pip:./"] + self.model_reqs
            )
        except ImportError:
            raise ValueError(
                "Could not import runhouse python package. "
                "Please install it with `pip install runhouse`."
            )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.client(self.model_name, False, texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.client(self.model_name, False, text)
        return embedding.tolist()


class SelfHostedHuggingFaceInstructEmbeddings(SelfHostedHuggingFaceEmbeddings):
    """Wrapper around InstructorEmbedding embedding models to perform inference on self-hosted remote hardware.
    Supported hardware includes auto-launched instances on AWS, GCP, Azure, and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another cloud like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import SelfHostedHuggingFaceInstructEmbeddings
            import runhouse as rh
            model_name = "hkunlp/instructor-large"
            gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')
            hf = SelfHostedHuggingFaceInstructEmbeddings(model_name=model_name, hardware=gpu)
    """

    client: Any  #: :meta private:
    model_name: str = DEFAULT_INSTRUCT_MODEL
    """Model name to use."""
    embed_instruction: str = DEFAULT_EMBED_INSTRUCTION
    """Instruction to use for embedding documents."""
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    """Instruction to use for embedding query."""
    model_reqs: List[str] = ["InstructorEmbedding"]
    """Requirements to install on hardware to inference the model."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace instruct model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        instruction_pairs = []
        for text in texts:
            instruction_pairs.append([self.embed_instruction, text])
        embeddings = self.client(self.model_name, True, instruction_pairs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace instruct model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        instruction_pair = [self.query_instruction, text]
        embedding = self.client(self.model_name, True, [instruction_pair])[0]
        return embedding.tolist()
