from enum import Enum
from typing import Any, List, Optional, Set, Union

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel


class TakeoffEmbeddingException(Exception):
    """Custom exception for interfacing with Takeoff Embedding class."""


class MissingConsumerGroup(TakeoffEmbeddingException):
    """Exception raised when no consumer group is provided on initialization of
    TitanTakeoffEmbed or in embed request."""


class Device(str, Enum):
    """Device to use for inference, cuda or cpu."""

    cuda = "cuda"
    cpu = "cpu"


class ReaderConfig(BaseModel):
    """Configuration for the reader to be deployed in Takeoff."""

    class Config:
        protected_namespaces = ()

    model_name: str
    """The name of the model to use"""

    device: Device = Device.cuda
    """The device to use for inference, cuda or cpu"""

    consumer_group: str = "primary"
    """The consumer group to place the reader into"""


class TitanTakeoffEmbed(Embeddings):
    """Interface with Takeoff Inference API for embedding models.

    Use it to send embedding requests and to deploy embedding
    readers with Takeoff.

    Examples:
        This is an example how to deploy an embedding model and send requests.

        .. code-block:: python
            # Import the TitanTakeoffEmbed class from community package
            import time
            from langchain_community.embeddings import TitanTakeoffEmbed

            # Specify the embedding reader you'd like to deploy
            reader_1 = {
                "model_name": "avsolatorio/GIST-large-Embedding-v0",
                "device": "cpu",
                "consumer_group": "embed"
            }

            # For every reader you pass into models arg Takeoff will spin up a reader
            # according to the specs you provide. If you don't specify the arg no models
            # are spun up and it assumes you have already done this separately.
            embed = TitanTakeoffEmbed(models=[reader_1])

            # Wait for the reader to be deployed, time needed depends on the model size
            # and your internet speed
            time.sleep(60)

            # Returns the embedded query, ie a List[float], sent to `embed` consumer
            # group where we just spun up the embedding reader
            print(embed.embed_query(
                "Where can I see football?", consumer_group="embed"
            ))

            # Returns a List of embeddings, ie a List[List[float]], sent to `embed`
            # consumer group where we just spun up the embedding reader
            print(embed.embed_document(
                ["Document1", "Document2"],
                consumer_group="embed"
            ))
    """

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
        """Initialize the Titan Takeoff embedding wrapper.

        Args:
            base_url (str, optional): The base url where Takeoff Inference Server is
            listening. Defaults to "http://localhost".
            port (int, optional): What port is Takeoff Inference API listening on.
            Defaults to 3000.
            mgmt_port (int, optional): What port is Takeoff Management API listening on.
            Defaults to 3001.
            models (List[ReaderConfig], optional): Any readers you'd like to spin up on.
            Defaults to [].

        Raises:
            ImportError: If you haven't installed takeoff-client, you will get an
            ImportError. To remedy run `pip install 'takeoff-client==0.4.0'`
        """
        self.base_url = base_url
        self.port = port
        self.mgmt_port = mgmt_port
        try:
            from takeoff_client import TakeoffClient
        except ImportError:
            raise ImportError(
                "takeoff-client is required for TitanTakeoff. "
                "Please install it with `pip install 'takeoff-client==0.4.0'`."
            )
        self.client = TakeoffClient(
            self.base_url, port=self.port, mgmt_port=self.mgmt_port
        )
        for model in models:
            self.client.create_reader(model)
            if isinstance(model, dict):
                self.embed_consumer_groups.add(model.get("consumer_group"))
            else:
                self.embed_consumer_groups.add(model.consumer_group)
        super(TitanTakeoffEmbed, self).__init__()

    def _embed(
        self, input: Union[List[str], str], consumer_group: Optional[str]
    ) -> dict:
        """Embed text.

        Args:
            input (List[str]): prompt/document or list of prompts/documents to embed
            consumer_group (Optional[str]): what consumer group to send the embedding
            request to. If not specified and there is only one
            consumer group specified during initialization, it will be used. If there
            are multiple consumer groups specified during initialization, you must
            specify which one to use.

        Raises:
            MissingConsumerGroup: The consumer group can not be inferred from the
            initialization and must be specified with request.

        Returns:
            Dict[str, Any]: Result of query, {"result": List[List[float]]} or
            {"result": List[float]}
        """
        if not consumer_group:
            if len(self.embed_consumer_groups) == 1:
                consumer_group = list(self.embed_consumer_groups)[0]
            elif len(self.embed_consumer_groups) > 1:
                raise MissingConsumerGroup(
                    "TakeoffEmbedding was initialized with multiple embedding reader"
                    "groups, you must specify which one to use."
                )
            else:
                raise MissingConsumerGroup(
                    "You must specify what consumer group you want to send embedding"
                    "response to as TitanTakeoffEmbed was not initialized with an "
                    "embedding reader."
                )
        return self.client.embed(input, consumer_group)

    def embed_documents(
        self, texts: List[str], consumer_group: Optional[str] = None
    ) -> List[List[float]]:
        """Embed documents.

        Args:
            texts (List[str]): List of prompts/documents to embed
            consumer_group (Optional[str], optional): Consumer group to send request
            to containing embedding model. Defaults to None.

        Returns:
            List[List[float]]: List of embeddings
        """
        return self._embed(texts, consumer_group)["result"]

    def embed_query(
        self, text: str, consumer_group: Optional[str] = None
    ) -> List[float]:
        """Embed query.

        Args:
            text (str): Prompt/document to embed
            consumer_group (Optional[str], optional): Consumer group to send request
            to containing embedding model. Defaults to None.

        Returns:
            List[float]: Embedding
        """
        return self._embed(text, consumer_group)["result"]
