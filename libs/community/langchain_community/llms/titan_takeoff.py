from enum import Enum
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import BaseModel

from langchain_community.llms.utils import enforce_stop_tokens


class Device(str, Enum):
    """The device to use for inference, cuda or cpu"""

    cuda = "cuda"
    cpu = "cpu"


class ReaderConfig(BaseModel):
    """Configuration for the reader to be deployed in Titan Takeoff API."""

    class Config:
        protected_namespaces = ()

    model_name: str
    """The name of the model to use"""

    device: Device = Device.cuda
    """The device to use for inference, cuda or cpu"""

    consumer_group: str = "primary"
    """The consumer group to place the reader into"""

    tensor_parallel: Optional[int] = None
    """The number of gpus you would like your model to be split across"""

    max_seq_length: int = 512
    """The maximum sequence length to use for inference, defaults to 512"""

    max_batch_size: int = 4
    """The max batch size for continuous batching of requests"""


class TitanTakeoff(LLM):
    """Titan Takeoff API LLMs.

    Titan Takeoff is a wrapper to interface with Takeoff Inference API for
    generative text to text language models.

    You can use this wrapper to send requests to a generative language model
    and to deploy readers with Takeoff.

    Examples:
        This is an example how to deploy a generative language model and send
        requests.

        .. code-block:: python
            # Import the TitanTakeoff class from community package
            import time
            from langchain_community.llms import TitanTakeoff

            # Specify the embedding reader you'd like to deploy
            reader_1 = {
                "model_name": "TheBloke/Llama-2-7b-Chat-AWQ",
                "device": "cuda",
                "tensor_parallel": 1,
                "consumer_group": "llama"
            }

            # For every reader you pass into models arg Takeoff will spin
            # up a reader according to the specs you provide. If you don't
            # specify the arg no models are spun up and it assumes you have
            # already done this separately.
            llm = TitanTakeoff(models=[reader_1])

            # Wait for the reader to be deployed, time needed depends on the
            # model size and your internet speed
            time.sleep(60)

            # Returns the query, ie a List[float], sent to `llama` consumer group
            # where we just spun up the Llama 7B model
            print(embed.invoke(
                "Where can I see football?", consumer_group="llama"
            ))

            # You can also send generation parameters to the model, any of the
            # following can be passed in as kwargs:
            # https://docs.titanml.co/docs/next/apis/Takeoff%20inference_REST_API/generate#request
            # for instance:
            print(embed.invoke(
                "Where can I see football?", consumer_group="llama", max_new_tokens=100
            ))
    """

    base_url: str = "http://localhost"
    """The base URL of the Titan Takeoff (Pro) server. Default = "http://localhost"."""

    port: int = 3000
    """The port of the Titan Takeoff (Pro) server. Default = 3000."""

    mgmt_port: int = 3001
    """The management port of the Titan Takeoff (Pro) server. Default = 3001."""

    streaming: bool = False
    """Whether to stream the output. Default = False."""

    client: Any = None
    """Takeoff Client Python SDK used to interact with Takeoff API"""

    def __init__(
        self,
        base_url: str = "http://localhost",
        port: int = 3000,
        mgmt_port: int = 3001,
        streaming: bool = False,
        models: List[ReaderConfig] = [],
    ):
        """Initialize the Titan Takeoff language wrapper.

        Args:
            base_url (str, optional): The base URL where the Takeoff
                Inference Server is listening. Defaults to `http://localhost`.
            port (int, optional): What port is Takeoff Inference API
                listening on. Defaults to 3000.
            mgmt_port (int, optional): What port is Takeoff Management API
                listening on. Defaults to 3001.
            streaming (bool, optional): Whether you want to by default use the
                generate_stream endpoint over generate to stream responses.
                Defaults to False. In reality, this is not significantly different
                as the streamed response is buffered and returned similar to the
                non-streamed response, but the run manager is applied per token
                generated.
            models (List[ReaderConfig], optional): Any readers you'd like to
                spin up on. Defaults to [].

        Raises:
            ImportError: If you haven't installed takeoff-client, you will
            get an ImportError. To remedy run `pip install 'takeoff-client==0.4.0'`
        """
        super().__init__(  # type: ignore[call-arg]
            base_url=base_url, port=port, mgmt_port=mgmt_port, streaming=streaming
        )
        try:
            from takeoff_client import TakeoffClient
        except ImportError:
            raise ImportError(
                "takeoff-client is required for TitanTakeoff. "
                "Please install it with `pip install 'takeoff-client>=0.4.0'`."
            )
        self.client = TakeoffClient(
            self.base_url, port=self.port, mgmt_port=self.mgmt_port
        )
        for model in models:
            self.client.create_reader(model)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "titan_takeoff"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Titan Takeoff (Pro) generate endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager to use when streaming.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                model = TitanTakeoff()

                prompt = "What is the capital of the United Kingdom?"

                # Use of model(prompt), ie `__call__` was deprecated in LangChain 0.1.7,
                # use model.invoke(prompt) instead.
                response = model.invoke(prompt)

        """
        if self.streaming:
            text_output = ""
            for chunk in self._stream(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
            ):
                text_output += chunk.text
            return text_output

        response = self.client.generate(prompt, **kwargs)
        text = response["text"]

        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call out to Titan Takeoff (Pro) stream endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager to use when streaming.

        Yields:
            A dictionary like object containing a string token.

        Example:
            .. code-block:: python

                model = TitanTakeoff()

                prompt = "What is the capital of the United Kingdom?"
                response = model.stream(prompt)

                # OR

                model = TitanTakeoff(streaming=True)

                response = model.invoke(prompt)

        """
        response = self.client.generate_stream(prompt, **kwargs)
        buffer = ""
        for text in response:
            buffer += text.data
            if "data:" in buffer:
                # Remove the first instance of "data:" from the buffer.
                if buffer.startswith("data:"):
                    buffer = ""
                if len(buffer.split("data:", 1)) == 2:
                    content, _ = buffer.split("data:", 1)
                    buffer = content.rstrip("\n")
                # Trim the buffer to only have content after the "data:" part.
                if buffer:  # Ensure that there's content to process.
                    chunk = GenerationChunk(text=buffer)
                    buffer = ""  # Reset buffer for the next set of data.
                    yield chunk
                    if run_manager:
                        run_manager.on_llm_new_token(token=chunk.text)

        # Yield any remaining content in the buffer.
        if buffer:
            chunk = GenerationChunk(text=buffer.replace("</s>", ""))
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(token=chunk.text)
