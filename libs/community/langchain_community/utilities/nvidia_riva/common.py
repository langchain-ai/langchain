"""A common module for NVIDIA Riva that contains utilities used by any Riva Runnable."""
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union, cast

from langchain_core.pydantic_v1 import (
    AnyHttpUrl,
    BaseModel,
    Field,
    parse_obj_as,
    validator,
)

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


class SentinelT:  # pylint: disable=too-few-public-methods
    """An empty Sentinel type."""


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
