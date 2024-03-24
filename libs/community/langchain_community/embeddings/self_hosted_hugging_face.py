import importlib
import logging
import runhouse as rh
from typing import Any, List, Optional

from langchain_core.pydantic_v1 import Extra

from langchain_community.embeddings.self_hosted import SelfHostedEmbeddings

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_TASK = "sentence-similarity"
DEFAULT_INSTRUCT_MODEL = "hkunlp/instructor-large"
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)

logger = logging.getLogger(__name__)


class TextModelEmbedding:
    def __init__(self, model_id: str, instruct: bool = False, device: int = 0):
        super().__init__()
        self.model_id, self.instruct, self.device = model_id, instruct, device
        self.client = None

    def load_embedding_model(self) -> Any:
        """Load the embedding model."""
        if not self.instruct:
            import sentence_transformers

            self.client = sentence_transformers.SentenceTransformer(self.model_id)
        else:
            from InstructorEmbedding import INSTRUCTOR

            self.client = INSTRUCTOR(self.model_id)

        if importlib.util.find_spec("torch") is not None:
            import torch

            cuda_device_count = torch.cuda.device_count()
            if self.device < -1 or (self.device >= cuda_device_count):
                raise ValueError(
                    f"Got device=={self.device}, "
                    f"device is required to be within [-1, {cuda_device_count})"
                )
            if self.device < 0 and cuda_device_count > 0:
                logger.warning(
                    "Device has %d GPUs available. "
                    "Provide device={deviceId} to `from_model_id` to use available"
                    "GPUs for execution. deviceId is -1 for CPU and "
                    "can be a positive integer associated with CUDA device id.",
                    cuda_device_count,
                )

            self.client = self.client.to(self.device)

    def embed_documents(self, *args: Any, **kwargs: Any) -> List[List[float]]:
        """Inference function to send to the remote hardware.

        Accepts a sentence_transformer model_id and
        returns a list of embeddings for each document in the batch.
        """
        return self.client.encode(*args, **kwargs)


class SelfHostedHuggingFaceEmbeddings(SelfHostedEmbeddings):
    """HuggingFace embedding models on self-hosted remote hardware.

    Supported hardware includes auto-launched instances on AWS, GCP, Azure,
    and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another cloud
    like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import SelfHostedHuggingFaceEmbeddings
            import runhouse as rh
            gpu = rh.cluster(name='rh-a10x', instance_type='g5.4xlarge', provider='aws')
            gpu.run(commands=["pip install langchain"])
            embedding_env = rh.env(name="embeddings_env",
                                   reqs=["transformers",
                                        "torch",
                                        "accelerate",
                                        "huggingface-hub",
                                        "sentence_transformers"],
                                   secrets=["huggingface"]
                                   # need for downloading models from huggingface
                                  ).to(system=gpu)
            hf = SelfHostedHuggingFaceEmbeddings(hardware=gpu, env=embedding_env)
    """

    client: Any  #: :meta private:
    model_id: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    env: Any
    """Env that will be sent to the remote hardware, 
    includes all requirements to install on hardware."""
    hardware: Any
    """Remote hardware to send the inference function to."""
    load_fn_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model load function."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any):
        """Initialize the remote inference function."""
        load_fn_kwargs = kwargs.pop("load_fn_kwargs", {})
        load_fn_kwargs["model_id"] = load_fn_kwargs.get("model_id", DEFAULT_MODEL_NAME)
        load_fn_kwargs["instruct"] = load_fn_kwargs.get("instruct", False)
        load_fn_kwargs["device"] = load_fn_kwargs.get("device", 0)
        load_fn_kwargs["hardware"] = load_fn_kwargs.get("hardware", None)
        load_fn_kwargs["env"] = load_fn_kwargs.get("device", None)
        load_fn_kwargs["task"] = load_fn_kwargs.get("task", DEFAULT_TASK)
        super().__init__(
            load_fn_kwargs=load_fn_kwargs, embeddings_cls=TextModelEmbedding, **kwargs
        )


class SelfHostedHuggingFaceInstructEmbeddings(SelfHostedHuggingFaceEmbeddings):

    """HuggingFace InstructEmbedding models on self-hosted remote hardware.

    Supported hardware includes auto-launched instances on AWS, GCP, Azure,
    and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another
    cloud like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import
                                                SelfHostedHuggingFaceInstructEmbeddings
            import runhouse as rh
            gpu = rh.cluster(name='rh-a10x', instance_type='g5.4xlarge', provider='aws')
            gpu.run(commands=["pip install langchain"])
            embedding_env = rh.env(name="embeddings_env",
                                   reqs=["transformers",
                                        "torch",
                                        "accelerate",
                                        "huggingface-hub",
                                        "sentence_transformers"],
                                   secrets=["huggingface"]
                                   # need for downloading models from huggingface
                                  ).to(system=gpu)
            hf = SelfHostedHuggingFaceInstructEmbeddings(hardware=gpu,
                                                         env=embedding_env)
    """  # noqa: E501

    model_id: str = DEFAULT_INSTRUCT_MODEL
    """Model name to use."""
    embed_instruction: str = DEFAULT_EMBED_INSTRUCTION
    """Instruction to use for embedding documents."""
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    """Instruction to use for embedding query."""
    env: rh.Env
    """Env that will be sent to the remote hardware, 
    includes all requirements to install on hardware."""
    hardware: rh.Cluster
    """Remote hardware to send the inference function to."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any):
        """Initialize the remote inference function."""
        load_fn_kwargs = kwargs.pop("load_fn_kwargs", {})
        load_fn_kwargs["model_id"] = load_fn_kwargs.get(
            "model_id", DEFAULT_INSTRUCT_MODEL
        )
        load_fn_kwargs["instruct"] = load_fn_kwargs.get("instruct", True)
        load_fn_kwargs["device"] = load_fn_kwargs.get("device", 0)
        load_fn_kwargs["task"] = load_fn_kwargs.get("task", DEFAULT_TASK)
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
        embeddings = self.EmbeddingsPipeline_remote_instance.embed_documents(
            instruction_pairs
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace instruct model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        instruction_pair = [self.query_instruction, text]
        embedding = self.EmbeddingsPipeline_remote_instance.embed_documents(
            instruction_pair
        )[0]
        return embedding.tolist()
