"""A collection of common code used by the NVIDIA Riva Runnables."""
import asyncio
import queue
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Generic,
    Iterator,
    Optional,
    TypeVar,
    Union,
    cast,
)

from langchain.pydantic_v1 import (
    AnyHttpUrl,
    BaseModel,
    Field,
    parse_obj_as,
    root_validator,
    validator,
)
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from pydantic import AnyUrl as AnyUrlv2

if TYPE_CHECKING:
    import riva.client


def _import_riva_client() -> "riva.client":
    """Import the riva client and raise an error on failure."""
    try:
        # pylint: disable-next=import-outside-toplevel # this client library is optional
        import riva.client
    except ImportError as err:
        raise ImportError(
            "Could not import the NVIDIA Riva client library. "
            "Please install it with `pip install nvidia-riva-client`."
        ) from err
    return riva.client


# Create a generic Sentinel type
class SentinelT:  # pylint: disable=too-few-public-methods
    """An empty Sentinel type."""


_TRANSFORM_END = SentinelT()


# Define base Riva data types
_InputT = TypeVar("_InputT")
_OutputT = TypeVar("_OutputT")


# pylint: disable-next=too-few-public-methods
class RivaBase(Generic[_InputT, _OutputT], RunnableSerializable[_InputT, _OutputT]):
    """A common set of methods for all Riva Runnables."""

    @root_validator(pre=True)
    @classmethod
    def _validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the Python environment and input arguments."""
        _ = _import_riva_client()
        return values

    async def atransform(
        self,
        input: AsyncIterator[_InputT],
        _: Optional[RunnableConfig] = None,
    ) -> AsyncGenerator[_OutputT, None]:
        """Intercept async transforms and route them to the synchronous transform in a thread."""
        loop = asyncio.get_running_loop()
        input_queue: queue.Queue[Union[_InputT, SentinelT]] = queue.Queue()
        out_queue: asyncio.Queue[Union[_OutputT, SentinelT]] = asyncio.Queue()

        async def _producer() -> None:
            """Produce input into the input queue."""
            async for val in input:
                input_queue.put(val)
            input_queue.put(_TRANSFORM_END)

        def _consumer() -> None:
            """Consume the input with transform."""
            input_iterator = cast(Iterator[_InputT], iter(input_queue.get, _TRANSFORM_END))
            for val in self.transform(input_iterator):
                out_queue.put_nowait(val)
            out_queue.put_nowait(_TRANSFORM_END)

        async def _consumer_coro() -> None:
            """Coroutine that wraps the consumer."""
            await loop.run_in_executor(None, _consumer)

        loop.create_task(_producer())
        loop.create_task(_consumer_coro())

        while True:
            try:
                val = await asyncio.wait_for(out_queue.get(), 0.5)
            except asyncio.exceptions.TimeoutError:
                continue
            out_queue.task_done()

            if val is _TRANSFORM_END:
                break

            yield val


class RivaAudioEncoding(str, Enum):
    """An enum of the possible choices for Riva audio encoding.

    The list of types exposed by the Riva GRPC Protobuf files can be found with the following commands:
    ```python
    import riva.client
    print(riva.client.AudioEncoding.keys())
    ```
    """

    ALAW = "ALAW"
    ENCODING_UNSPECIFIED = "ENCODING_UNSPECIFIED"
    FLAC = "FLAC"
    LINEAR_PCM = "LINEAR_PCM"
    MULAW = "MULAW"
    OGGOPUS = "OGGOPUS"

    @classmethod
    def from_wave_format_code(cls, format_code: int) -> "RivaAudioEncoding":
        """Return the audio encoding specified by the format code in the wave file.
        ref: https://mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
        """
        try:
            return {1: cls.LINEAR_PCM, 6: cls.ALAW, 7: cls.MULAW}[format_code]
        except KeyError as err:
            raise NotImplementedError(
                f"The following wave file format code is not supported by Riva: {format_code}"
            ) from err

    @property
    def riva_pb2(self) -> "riva.client.AudioEncoding":
        """Returns the Riva API object for the encoding."""
        riva_client = _import_riva_client()
        return getattr(riva_client.AudioEncoding, self)


class RivaAuthMixin(BaseModel):
    """Configuration for the authentication to a Riva service connection."""

    url: Union[AnyHttpUrl, str] = Field(
        AnyHttpUrl("http://localhost:50051", scheme="http"),
        description="The full URL where the Riva service can be found.",
        examples=["http://localhost:50051", "https://user@pass:riva.example.com"],
    )
    ssl_cert: Optional[str] = Field(
        None,
        description="A full path to the file where Riva's public ssl key can be read.",
    )

    @property
    def auth(self) -> "riva.client.Auth":
        """Return a riva client auth object."""
        riva_client = _import_riva_client()
        url = cast(AnyHttpUrl, self.url)
        use_ssl = url.scheme == "https"  # pylint: disable=no-member # false positive
        url_no_scheme = str(self.url).split("/")[2]
        return riva_client.Auth(
            ssl_cert=self.ssl_cert, use_ssl=use_ssl, uri=url_no_scheme
        )

    @validator("url", pre=True)
    @classmethod
    def _validate_url(cls, val: Any) -> AnyHttpUrl:
        """Do some initial conversations for the URL before checking."""
        if isinstance(val, AnyUrlv2):
            return cast(AnyHttpUrl, AnyHttpUrl(str(val), scheme=val.scheme))
        if isinstance(val, str):
            return cast(AnyHttpUrl, parse_obj_as(AnyHttpUrl, val))
        return cast(AnyHttpUrl, val)


class RivaCommonConfigMixin(BaseModel):
    """A collection of common Riva settings."""

    encoding: RivaAudioEncoding = Field(
        default=RivaAudioEncoding.LINEAR_PCM,
        description="The encoding on the audio stream.",
    )
    sample_rate_hertz: int = Field(
        default=8000, description="The sample rate frequency of audio stream."
    )
    language_code: str = Field(
        default="en-US",
        description="The [BCP-47 language code](https://www.rfc-editor.org/rfc/bcp/bcp47.txt) for the target lanauge.",
    )
