"""Wrapper around HuggingFace embedding models for self-hosted remote hardware."""
import importlib
import logging
from typing import Any, Callable, List, Optional

from langchain.embeddings.self_hosted import SelfHostedEmbeddings

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_INSTRUCT_MODEL = "hkunlp/instructor-large"
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)

logger = logging.getLogger(__name__)


def _embed_documents(client: Any, *args: Any, **kwargs: Any) -> List[List[float]]:
    """Inference function to send to the remote hardware.

    Accepts a sentence_transformer model_id and
    returns a list of embeddings for each document in the batch.
    """
    return client.encode(*args, **kwargs)


def load_embedding_model(model_id: str, instruct: bool = False, device: int = 0) -> Any:
    """Load the embedding model."""
    if not instruct:
        import sentence_transformers

        client = sentence_transformers.SentenceTransformer(model_id)
    else:
        from InstructorEmbedding import INSTRUCTOR

        client = INSTRUCTOR(model_id)

    if importlib.util.find_spec("torch") is not None:
        import torch

        cuda_device_count = torch.cuda.device_count()
        if device < -1 or (device >= cuda_device_count):
            raise ValueError(
                f"Got device=={device}, "
                f"device is required to be within [-1, {cuda_device_count})"
            )
        if device < 0 and cuda_device_count > 0:
            logger.warning(
                "Device has %d GPUs available. "
                "Provide device={deviceId} to `from_model_id` to use available"
                "GPUs for execution. deviceId is -1 for CPU and "
                "can be a positive integer associated with CUDA device id.",
                cuda_device_count,
            )

        client = client.to(device)
    return client


class SelfHostedHuggingFaceEmbeddings(SelfHostedEmbeddings):
    """Runs sentence_transformers embedding models on self-hosted remote hardware.

    Supported hardware includes auto-launched instances on AWS, GCP, Azure,
    and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another cloud
    like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import SelfHostedHuggingFaceEmbeddings
            import runhouse as rh
            model_name = "sentence-transformers/all-mpnet-base-v2"
            gpu = rh.cluster(name="rh-a10x", instance_type="A100:1")
            hf = SelfHostedHuggingFaceEmbeddings(model_name=model_name, hardware=gpu)
    """

    client: Any  #: :meta private:
    model_id: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    model_reqs: List[str] = ["./", "sentence_transformers", "torch"]
    """Requirements to install on hardware to inference the model."""
    hardware: Any
    """Remote hardware to send the inference function to."""
    model_load_fn: Callable = load_embedding_model
    """Function to load the model remotely on the server."""
    load_fn_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model load function."""
    inference_fn: Callable = _embed_documents
    """Inference function to extract the embeddings."""

    def __init__(self, **kwargs: Any):
        """Initialize the remote inference function."""
        load_fn_kwargs = kwargs.pop("load_fn_kwargs", {})
        load_fn_kwargs["model_id"] = load_fn_kwargs.get("model_id", DEFAULT_MODEL_NAME)
        load_fn_kwargs["instruct"] = load_fn_kwargs.get("instruct", False)
        load_fn_kwargs["device"] = load_fn_kwargs.get("device", 0)
        super().__init__(load_fn_kwargs=load_fn_kwargs, **kwargs)


class SelfHostedHuggingFaceInstructEmbeddings(SelfHostedHuggingFaceEmbeddings):
    """Runs InstructorEmbedding embedding models on self-hosted remote hardware.

    Supported hardware includes auto-launched instances on AWS, GCP, Azure,
    and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another
    cloud like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import SelfHostedHuggingFaceInstructEmbeddings
            import runhouse as rh
            model_name = "hkunlp/instructor-large"
            gpu = rh.cluster(name='rh-a10x', instance_type='A100:1')
            hf = SelfHostedHuggingFaceInstructEmbeddings(
                model_name=model_name, hardware=gpu)
    """

    model_id: str = DEFAULT_INSTRUCT_MODEL
    """Model name to use."""
    embed_instruction: str = DEFAULT_EMBED_INSTRUCTION
    """Instruction to use for embedding documents."""
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    """Instruction to use for embedding query."""
    model_reqs: List[str] = ["./", "InstructorEmbedding", "torch"]
    """Requirements to install on hardware to inference the model."""

    def __init__(self, **kwargs: Any):
        """Initialize the remote inference function."""
        load_fn_kwargs = kwargs.pop("load_fn_kwargs", {})
        load_fn_kwargs["model_id"] = load_fn_kwargs.get(
            "model_id", DEFAULT_INSTRUCT_MODEL
        )
        load_fn_kwargs["instruct"] = load_fn_kwargs.get("instruct", True)
        load_fn_kwargs["device"] = load_fn_kwargs.get("device", 0)
        super().__init__(load_fn_kwargs=load_fn_kwargs, **kwargs)

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
        embeddings = self.client(self.pipeline_ref, instruction_pairs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace instruct model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        instruction_pair = [self.query_instruction, text]
        embedding = self.client(self.pipeline_ref, [instruction_pair])[0]
        return embedding.tolist()
