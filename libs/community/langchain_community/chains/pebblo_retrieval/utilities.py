import logging
import os
import platform
from enum import Enum
from typing import Any, Optional, Tuple

from langchain_core.env import get_runtime_environment
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils import get_from_dict_or_env

from langchain_community.chains.pebblo_retrieval.models import Framework, Runtime

logger = logging.getLogger(__name__)

PLUGIN_VERSION = "0.1.1"

CLASSIFIER_URL = os.getenv("PEBBLO_CLASSIFIER_URL", "http://localhost:8000")
PEBBLO_CLOUD_URL = os.getenv("PEBBLO_CLOUD_URL", "https://api.daxa.ai")
_DEFAULT_CLASSIFIER_URL = "http://localhost:8000"
_DEFAULT_PEBBLO_CLOUD_URL = "https://api.daxa.ai"

PROMPT_URL = "/v1/prompt"
PROMPT_GOV_URL = "/v1/prompt/governance"
APP_DISCOVER_URL = "/v1/app/discover"
class Routes(str, Enum):
    """Routes available for the Pebblo API as enumerator."""

    retrieval_app_discover = "/v1/app/discover"
    prompt = "/v1/prompt"
    prompt_governance = "/v1/prompt/governance"


def get_runtime() -> Tuple[Framework, Runtime]:
    """Fetch the current Framework and Runtime details.

    Returns:
        Tuple[Framework, Runtime]: Framework and Runtime for the current app instance.
    """
    runtime_env = get_runtime_environment()
    framework = Framework(name="langchain", version=runtime_env.get("library_version"))
    uname = platform.uname()
    runtime = Runtime(
        host=uname.node,
        path=os.environ["PWD"],
        platform=runtime_env.get("platform", "unknown"),
        os=uname.system,
        os_version=uname.version,
        ip=get_ip(),
        language=runtime_env.get("runtime", "unknown"),
        language_version=runtime_env.get("runtime_version", "unknown"),
    )

    if "Darwin" in runtime.os:
        runtime.type = "desktop"
        runtime.runtime = "Mac OSX"

    logger.debug(f"framework {framework}")
    logger.debug(f"runtime {runtime}")
    return framework, runtime


def get_ip() -> str:
    """Fetch local runtime ip address.

    Returns:
        str: IP address
    """
    import socket  # lazy imports

    host = socket.gethostname()
    try:
        public_ip = socket.gethostbyname(host)
    except Exception:
        public_ip = socket.gethostbyname("localhost")
    return public_ip


class PebbloAPIWrapper(BaseModel):
    """Wrapper for Pebblo API."""

    api_key: Optional[str]  # Use SecretStr
    """API key for Pebblo Cloud"""
    classifier_location: str = "local"
    """Location of the classifier, local or cloud. Defaults to 'local'"""
    classifier_url: Optional[str]
    """URL of the Pebblo Classifier"""
    cloud_url: Optional[str]
    """URL of the Pebblo Cloud"""

    def __init__(self, **kwargs: Any):
        """Validate that api key in environment."""
        kwargs["api_key"] = get_from_dict_or_env(
            kwargs, "api_key", "PEBBLO_API_KEY", ""
        )
        kwargs["classifier_url"] = get_from_dict_or_env(
            kwargs, "classifier_url", "PEBBLO_CLASSIFIER_URL", _DEFAULT_CLASSIFIER_URL
        )
        kwargs["cloud_url"] = get_from_dict_or_env(
            kwargs, "cloud_url", "PEBBLO_CLOUD_URL", _DEFAULT_PEBBLO_CLOUD_URL
        )
        super().__init__(**kwargs)
