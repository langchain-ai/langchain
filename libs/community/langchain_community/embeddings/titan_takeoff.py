from typing import Any, List, Optional, Set

import logging
from pydantic import BaseModel
from enum import Enum

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import GenerationChunk

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__file__)


class TakeoffEmbeddingException(Exception):
    """Exceptions experienced with interfacing with Takeoff Embedding Wrapper"""


class MissingConsumerGroup(TakeoffEmbeddingException):
    """Exception raised when no consumer group is provided on initialization of TitanTakeoffEmbed or in embed request"""


class Device(str, Enum):
    """The device to use for inference, cuda or cpu"""

    cuda = "cuda"
    cpu = "cpu"


class ReaderConfig(BaseModel):
    class Config:
        protected_namespaces = ()

    model_name: str
    """The name of the model to use"""

    device: Device = Device.cuda
    """The device to use for inference, cuda or cpu"""

    consumer_group: str = "primary"
    """The consumer group to place the reader into"""


class TitanTakeoffEmbed(Embeddings):
    """Titan Takeoff Pro is a language model that can be used to generate text."""

    base_url: str = "http://localhost"
    """The base URL of the Titan Takeoff (Pro) server. Default = "http://localhost"."""

    port: int = 3000
    """The port of the Titan Takeoff (Pro) server. Default = 3000."""

    mgmt_port: int = 3001
    """The management port of the Titan Takeoff (Pro) server. Default = 3001."""

    client: Any = None
    """Takeoff Client Python SDK used to interact with Takeoff API"""

    embed_consumer_groups: Set[str] = set()
    """The consumer groups in Takeoff which contain embedding models"""

    def __init__(
        self,
        base_url: str = "http://localhost",
        port: int = 3000,
        mgmt_port: int = 3001,
        models: List[ReaderConfig] = [],
    ):
        """Initialize the Titan Takeoff Pro language model."""
        self.base_url = base_url
        self.port = port
        self.mgmt_port = mgmt_port
        try:
            from takeoff_client import TakeoffClient
        except ImportError:
            raise ImportError(
                "takeoff-client is required for TitanTakeoff. " "Please install it with `pip install 'takeoff-client>=0.4.0'`."
            )
        self.client = TakeoffClient(self.base_url, port=self.port, mgmt_port=self.mgmt_port)
        for model in models:
            self.client.create_reader(model)
            self.embed_consumer_groups.add(model["consumer_group"])
        super(TitanTakeoffEmbed, self).__init__()
        
    def _embed(self, input: List[str], consumer_group: Optional[str]) -> dict[str, Any]:
        """Embed a list of strings."""
        if not consumer_group:
            if len(self.embed_consumer_groups) == 1:
                consumer_group = list(self.embed_consumer_groups)[0]
            elif len(self.embed_consumer_groups) > 1:
                raise MissingConsumerGroup("TakeoffEmbedding was initialized with multiple embedding reader groups, you must specify which one to use.")
            else:
                raise MissingConsumerGroup(
                    "You must specify what consumer group you want to send embedding response to as TitanTakeoffEmbed was not initialized with"
                    " an embedding reader."
                )
        return self.client.embed(input, consumer_group)

    def embed_documents(self, texts: List[str], consumer_group: Optional[str] = None) -> List[List[float]]:
        """Embed search docs."""
        return self._embed(texts, consumer_group)["text"]

    def embed_query(self, text: str, consumer_group: Optional[str] = None) -> List[float]:
        """Embed query text."""
        return self._embed(text, consumer_group)["text"]
