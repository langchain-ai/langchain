import logging
import os
import platform
from typing import Tuple

from langchain_core.env import get_runtime_environment

from langchain_community.chains.pebblo_retrieval.models import Framework, Runtime

logger = logging.getLogger(__name__)

PLUGIN_VERSION = "0.1.1"

CLASSIFIER_URL = os.getenv("PEBBLO_CLASSIFIER_URL", "http://localhost:8000")
PEBBLO_CLOUD_URL = os.getenv("PEBBLO_CLOUD_URL", "https://api.daxa.ai")

PROMPT_URL = "/v1/prompt"
PROMPT_GOV_URL = "/v1/prompt/governance"
APP_DISCOVER_URL = "/v1/app/discover"


def get_runtime() -> Tuple[Framework, Runtime]:
    """Fetch the current Framework and Runtime details.

    Returns:
        Tuple[Framework, Runtime]: Framework and Runtime for the current app instance.
    """
    runtime_env = get_runtime_environment()
    framework = Framework(
        name="langchain", version=runtime_env.get("library_version", None)
    )
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
