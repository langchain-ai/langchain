"""Wrapper for untrusted code exectuion on docker."""
# TODO:  pass payload to contanier via filesystem 
# TEST:  more tests for attach to running container
# TODO:  embed file payloads in the call to run (in LLMChain)?
# TODO:  [doc] image selection helper
# TODO:  LLMChain decorator ?


import docker
from typing import Any
import logging
logger = logging.getLogger(__name__)


GVISOR_WARNING = """Warning: gVisor runtime not available for {docker_host}.

Running untrusted code in a container without gVisor is not recommended. Docker
containers are not isolated. They can be abused to gain access to the host
system. To mitigate this risk, gVisor can be used to run the container in a
sandboxed environment. see: https://gvisor.dev/ for more info.
"""


def gvisor_runtime_available(client: Any) -> bool:
    """Verify if gVisor runtime is available."""
    logger.debug("verifying availability of gVisor runtime...")
    info = client.info()
    if 'Runtimes' in info:
        return 'runsc' in info['Runtimes']
    return False


def _check_gvisor_runtime():
    client = docker.from_env()
    docker_host = client.api.base_url
    if not gvisor_runtime_available(docker.from_env()):
        logger.warning(GVISOR_WARNING.format(docker_host=docker_host))


_check_gvisor_runtime()

from .tool import DockerWrapper
