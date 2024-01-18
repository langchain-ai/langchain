"""The definition of the application configuration."""
import logging
from enum import Enum
from typing import Optional

from confz import BaseConfig, EnvSource, FileSource
from pydantic import BaseModel, Field, field_validator

from .contributions.langchain_nv_riva import RivaAuthMixin

_ENV_VAR_PREFIX = "APP_"
_CONFIG_FILE_ENV_VAR: str = f"{_ENV_VAR_PREFIX}CONFIG"


class LogLevels(Enum):
    """An enumerator of all the logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


RivaService = RivaAuthMixin


class RivaASR(BaseModel):
    """Configuration for connecting to a Riva ASR service."""

    service: RivaService = Field(default_factory=RivaService)  # type: ignore
    profanity_filter: bool = Field(
        True,
        description="Controls whether or not Riva should attempt to filter profanity out of the transcribed text.",
    )
    enable_automatic_punctuation: bool = Field(
        True,
        description="Controls whether Riva should attempt to correct senetence puncuation in the transcribed text.",
    )
    language_code: str = Field(
        "en-US",
        description="The [BCP-47 language code](https://www.rfc-editor.org/rfc/bcp/bcp47.txt) for the target lanauge.",
    )


class RivaTTS(BaseModel):
    """Configuration for connecting to a Riva TTS service."""

    service: RivaService = Field(default_factory=RivaService)  # type: ignore
    output_directory: Optional[str] = None
    language_code: str = "en-US"
    voice_name: str = "English-US.Female-1"


class Configuration(BaseConfig):
    """Configuration for this microservice.

    By default, configuration is looked for in the following locations. Later files will overwrite settings
    values from earlier files.

    **Default Config File Search Locations:**
    The configuration file will be searched for in all of the following locations.
    Values from lower files in the list will take precendence.
    - ./config.yaml
    - ./config.yml
    - ./config.json
    - ~/app.yaml
    - ~/app.yml
    - ~/app.json
    - /etc/app.yaml
    - /etc/app.yml
    - /etc/app.json

    **Custom Config File Search Locations:**
    An additional config file path can be specified through an environment variable.
    The value in this file will take precedence over the default files.
    ```bash
    export APP_CONFIG=/etc/my_config.yaml
    ```

    **Config From Environment Variables:**
    Configuration can also be set using environment variables.
    The variable names will be in the form: `APP_FIELD.SUB_FIELD`
    These will take precedence over any of the files.
    The following is an example variable:
    ```bash
    export APP_RIVA_ASR.SERVICE.URL="https://localhost:8443"
    ```
    """

    riva_asr: RivaASR = Field(default_factory=RivaASR)  # type: ignore
    riva_tts: RivaTTS = Field(default_factory=RivaTTS)
    log_level: LogLevels = LogLevels.WARNING

    CONFIG_SOURCES = [
        FileSource(file="./config.yaml", optional=True),
        FileSource(file="./config.yml", optional=True),
        FileSource(file="./config.json", optional=True),
        FileSource(file="~/app.yaml", optional=True),
        FileSource(file="~/app.yml", optional=True),
        FileSource(file="~/app.json", optional=True),
        FileSource(file="/etc/app.yaml", optional=True),
        FileSource(file="/etc/app.yml", optional=True),
        FileSource(file="/etc/app.json", optional=True),
        FileSource(file_from_env=_CONFIG_FILE_ENV_VAR, optional=True),
        EnvSource(allow_all=True, prefix=_ENV_VAR_PREFIX),
    ]

    @field_validator("log_level", mode="after")
    @classmethod
    def _check_log_level(cls, val: LogLevels) -> LogLevels:
        """Configure the default loggers."""
        logging.basicConfig()
        log_level = logging.getLevelName(val.value)
        loggers = [logging.getLogger()] + [
            logging.getLogger(name)
            for name in logging.root.manager.loggerDict  # pylint: disable=no-member
        ]
        for logger in loggers:
            logger.setLevel(log_level)
        return val


# load the runtime configuration
config = Configuration()
