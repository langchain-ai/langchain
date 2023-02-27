"""Wrapper for untrusted code exectuion on docker."""
# TODO: Validation:
#       - verify gVisor runtime (runsc) if available
#	    -TEST: spawned container: make sure it's ready ? before sending/reading commands
#       - attach to running container
#       - pass arbitrary image names
#		- embed file payloads in the call to run (in LLMChain)?
#       - image selection helper
#       - LLMChain decorator ?

import docker
import struct
import pandas as pd # type: ignore
from docker.client import DockerClient  # type: ignore
from docker.errors import APIError, ContainerError # type: ignore
import logging

from typing import Any, Dict
from typing import Optional
from pydantic import BaseModel, PrivateAttr, Extra, root_validator, validator

logger = logging.getLogger(__name__)

docker_images = {
        'default': 'alpine:{version}',
        'python': 'python:{version}',
        }

SOCK_BUF_SIZE = 1024

GVISOR_WARNING = """Warning: gVisor runtime not available for {docker_host}.

Running untrusted code in a container without gVisor is not recommended. Docker
containers are not isolated. They can be abused to gain access to the host
system. To mitigate this risk, gVisor can be used to run the container in a
sandboxed environment. see: https://gvisor.dev/ for more info.
"""

def gvisor_runtime_available(client: DockerClient) -> bool:
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

class DockerSocket:
    """Wrapper around docker API's socket object. Can be used as a context manager."""


    def __init__(self, socket):
        self.socket = socket
        # self.socket._sock.setblocking(False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        logger.debug("closing socket...")
        self.socket._sock.shutdown(2) # 2 = SHUT_RDWR
        self.socket._sock.close()
        self.socket.close()

    def sendall(self, data: bytes) -> None:
        self.socket._sock.sendall(data)

    def recv(self):
        """Wrapper for socket.recv that does buffured read."""

        # NOTE: this is optional as a bonus
        # TODO: Recv with TTY enabled
        #
        # When the TTY setting is enabled in POST /containers/create, the stream
        # is not multiplexed. The data exchanged over the hijacked connection is
        # simply the raw data from the process PTY and client's stdin.



        # header := [8]byte{STREAM_TYPE, 0, 0, 0, SIZE1, SIZE2, SIZE3, SIZE4}
        # STREAM_TYPE can be:
        #
        # 0: stdin (is written on stdout)
        # 1: stdout
        # 2: stderr
        # SIZE1, SIZE2, SIZE3, SIZE4 are the four bytes of the uint32 size encoded as big endian.
        #
        # Following the header is the payload, which is the specified number of bytes of STREAM_TYPE.
        #
        # The simplest way to implement this protocol is the following:
        #
        # - Read 8 bytes.
        # - Choose stdout or stderr depending on the first byte.
        # - Extract the frame size from the last four bytes.
        # - Read the extracted size and output it on the correct output.
        # - Goto 1.

        chunks = []
        # try:
        #     self.socket._sock.recv(8)
        # except BlockingIOError as e:
        #     raise ValueError("incomplete read from container output")

        while True:
            header = b''
            try:
                # strip the header
                # the first recv is blocking to wait for the container to start
                header = self.socket._sock.recv(8)
            except BlockingIOError:
                print("[header] blocking IO")
                break

            self.socket._sock.setblocking(False)

            if header == b'':
                break
            stream_type, size = struct.unpack("!BxxxI", header)

            payload = b''
            while size:
                chunk = b''
                try:
                    chunk = self.socket._sock.recv(min(size, SOCK_BUF_SIZE))
                except BlockingIOError:
                    print("[body] blocking IO")
                    break
                if chunk == b'':
                    raise ValueError("incomplete read from container output")
                payload += chunk
                size -= len(chunk)
            chunks.append((stream_type, payload))
            # try:
            #     msg = self.socket._sock.recv(SOCK_BUF_SIZE)
            #     chunk += msg
            # except BlockingIOError as e:
            #     break

        return chunks




class DockerWrapper(BaseModel, extra=Extra.forbid):
    """Executes arbitrary payloads and returns the output."""

    _docker_client: DockerClient = PrivateAttr()
    image: Optional[str] = "alpine"

    # use env by default when create docker client
    from_env: Optional[bool] = True

    def __init__(self, **kwargs):
        """Initialize docker client."""
        super().__init__(**kwargs)

        if self.from_env:
            self._docker_client = docker.from_env()


    @property
    def client(self) -> DockerClient:
        """Docker client."""
        return self._docker_client

    @property
    def info(self) -> Any:
        """Prints docker `info`."""
        return self._docker_client.info()

    @root_validator()
    def validate_all(cls, values: Dict) -> Dict:
        """Validate environment."""
        return values

    def run(self, query: str, **kwargs: Any) -> str:
        """Run arbitrary shell command inside a container.

        Args:
            **kwargs: Pass extra parameters to DockerClient.container.run.

        """
        try:
            image = kwargs.get("image", self.image)
            return self._docker_client.containers.run(image,
                                                      query,
                                                      remove=True,
                                                      **kwargs)
        except ContainerError as e:
            return f"STDERR: {e}"

        # TODO: handle docker APIError ?
        except APIError as e:
            print(f"APIError: {e}")
            return "ERROR"



    def exec_run(self, query: str, image: str) -> str:
        """Run arbitrary shell command inside a container.

        This is a lower level API that sends the input to the container's
        stdin through a socket using Docker API. It effectively simulates a tty session.
        """
        # it is necessary to open stdin to keep the container running after it's started
        # the attach_socket will hold the connection open until the container is stopped or
        # the socket is closed.

        # TODO!: use tty=True to be able to simulate a tty session.

        # NOTE: some images like python need to be launched with custom
        # parameters to keep stdin open. For example python image needs to be
        # started with the command `python3 -i`

        # TODO: handle both output mode for tty=True/False
        container = self._docker_client.containers.create(image, stdin_open=True)
        container.start()
        # input()

        # get underlying socket
        _socket = container.attach_socket(params={'stdout': 1, 'stderr': 1,
                                                  'stdin': 1, 'stream': 1})

        with DockerSocket(_socket) as socket:
            # TEST: make sure the container is ready ? use a blocking call first 
            socket.sendall(query.encode('utf-8'))

            # read the output
            output = None
            output = socket.recv()
            # print(output)



        container.kill()
        container.remove()

        # output is stored in a list of tuples (stream_type, payload)
        df = pd.DataFrame(output, columns=['stream_type', 'payload'])
        df['payload'] = df['payload'].apply(lambda x: x.decode('utf-8'))
        df['stream_type'] = df['stream_type'].apply(lambda x: 'stdout' if x == 1 else 'stderr')
        payload = df.groupby('stream_type')['payload'].apply(''.join).to_dict()
        print(payload)

        if 'stdout' in payload and 'stderr' in payload:
            return f"STDOUT:\n {payload['stdout']}\nSTDERR: {payload['stderr']}"
        elif 'stderr' in payload:
            return f"STDERR: {payload['stderr']}"
        else:
            return payload['stdout']

