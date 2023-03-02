"""Wrapper for untrusted code exectuion on docker."""
#TODO:  attach to running container
#TODO:  pull images
#TODO:  embed file payloads in the call to run (in LLMChain)?
#TODO:  image selection helper
#TODO:  LLMChain decorator ?

import docker
import struct
import socket
import shlex
from time import sleep
import pandas as pd  # type: ignore
from docker.client import DockerClient  # type: ignore
from docker.errors import APIError, ContainerError  # type: ignore
import logging

from .images import BaseImage, get_image_template, Python, Shell

from typing import Any, Dict, Optional, Union, Type, List
from pydantic import BaseModel, PrivateAttr, Extra, root_validator, validator, Field

logger = logging.getLogger(__name__)

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

#TODO!: using pexpect to with containers
# TODO: add default expect pattern to image template
# TODO: pass max reads parameters for read trials
# NOTE: spawning with tty true or not gives slightly different stdout format
# NOTE: echo=False works when tty is disabled and only stdin is connected


class DockerSocket:
    """Wrapper around docker API's socket object. Can be used as a context manager."""

    _timeout: int = 5


    def __init__(self, socket, timeout: int = _timeout):
        self.socket = socket
        self.socket._sock.settimeout(timeout)
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

    def setblocking(self, flag: bool) -> None:
        self.socket._sock.setblocking(flag)

    def recv(self) -> Any:
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
                # logger.debug("[header] blocking IO")
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
                    # logger.debug("[body] blocking IO")
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


def _default_params() -> Dict:
    return {
            # the only required parameter to be able to attach.
            'stdin_open': True,
            }

def _get_command(query: str, **kwargs: Dict) -> str:
    """Build an escaped command from a query string and keyword arguments."""
    cmd = query
    if 'default_command' in kwargs:
        cmd = shlex.join([*kwargs.get('default_command'), query])  # type: ignore

    return cmd

class DockerWrapper(BaseModel, extra=Extra.allow):
    """Executes arbitrary payloads and returns the output.


    Args:
        image (str | Type[BaseImage]): Docker image to use for execution. The
        image can be a string or a subclass of images.BaseImage.
        default_command (List[str]): Default command to use when creating the container.
    """

    _docker_client: DockerClient = PrivateAttr()
    _params: Dict = Field(default_factory=Shell().dict(), skip=True)
    image: Union[str, Type[BaseImage]] = Field(default_factory=Shell,skip=True)
    from_env: Optional[bool] = Field(default=True, skip=True)


    # @property
    # def image_name(self) -> str:
    #     """The image name that will be used when creating a container."""
    #     return self._params.image
    #
    def __init__(self, **kwargs):
        """Initialize docker client."""
        super().__init__(**kwargs)

        if self.from_env:
            self._docker_client = docker.from_env()

        # if not isinstance(self.image, str) and issubclass(self.image, BaseImage):
        #     self._params = {**self._params, **self.image().dict()}
        #
        # # if the user defined a custom image not pre registerd already we should
        # # not use the custom command
        # elif isinstance(self.image, str):
        #     self._params = {**_default_params(), **{'image': self.image}}

    @property
    def client(self) -> DockerClient:
        """Docker client."""
        return self._docker_client

    @property
    def info(self) -> Any:
        """Prints docker `info`."""
        return self._docker_client.info()

    # @validator("image", pre=True, always=True)
    # def validate_image(cls, value):
    #     if value is None:
    #         raise ValueError("image is required")
    #     if isinstance(value, str) :
    #         image = get_image(value)
    #         if isinstance(image, BaseImage):
    #             return image
    #         else:
    #             #set default params to base ones
    #     if issubclass(value, BaseImage):
    #         return value
    #     else:
    #         raise ValueError("image must be a string or a subclass of images.BaseImage")

    @root_validator()
    def validate_all(cls, values: Dict) -> Dict:
        """Validate environment."""
        image = values.get("image")
        if image is None:
            raise ValueError("image is required")
        if isinstance(image, str):
            # try to get image
            _image = get_image_template(image)
            if isinstance(_image, str):
                # user wants a custom image, we should use default params
                values["_params"] = {**_default_params(), **{'image': image}}
            else:
                # user wants a pre registered image, we should use the image params
                values["_params"] = _image().dict()
        # image is a BaseImage class
        elif issubclass(image.__class__, BaseImage):
            values["_params"] = image.dict()


        def field_filter(x):
            fields = cls.__fields__
            if x[0] == '_params':
                return False
            field = fields.get(x[0], None)
            if not field:
                return True
            return not field.field_info.extra.get('skip', False)
        filtered_fields: Dict[Any, Any] = dict(filter(field_filter, values.items()))  # type: ignore
        values["_params"] = {**values["_params"],
                             **filtered_fields}

        return values

    def run(self, query: str, **kwargs: Any) -> str:
        """Run arbitrary shell command inside a container.

        This method will concatenate the registered default command with the provided
        query.

        Args:
            query (str): The command to run.
            **kwargs: Pass extra parameters to DockerClient.container.run.

        """
        kwargs = {**self._params, **kwargs}
        args = {
                'image': self._params.get('image'),
                'command': query,
                }

        del kwargs['image']
        cmd = _get_command(query, **kwargs)
        kwargs.pop('default_command', None)

        args['command'] = cmd
        # print(f"args: {args}")
        # print(f"kwargs: {kwargs}")
        # return
        logger.debug(f"running command {args['command']}")
        logger.debug(f"with params {kwargs}")
        try:
            result= self._docker_client.containers.run(*(args.values()),
                                                      remove=True,
                                                      **kwargs)
            return result.decode('utf-8').strip()
        except ContainerError as e:
            return f"STDERR: {e}"

        # TODO: handle docker APIError ?
        except APIError as e:
            logger.debug(f"APIError: {e}")
            return "ERROR"



    def exec_run(self, query: str, timeout: int = 5,
                 delay: float = 0.5,
                 with_stderr: bool = False,
                 **kwargs: Any) -> str:
        """Run arbitrary shell command inside an ephemeral container.

        This will create a container, run the command, and then remove the
        container. the input is sent to the container's stdin through a socket
        using Docker API. It effectively simulates a tty session.

        Args:
            timeout (int): The timeout for receiving from the attached stdin.
            delay (int): The delay in seconds before running the command.
            **kwargs: Pass extra parameters to DockerClient.container.exec_run.
        """
        # it is necessary to open stdin to keep the container running after it's started
        # the attach_socket will hold the connection open until the container is stopped or
        # the socket is closed.

        # NOTE: using tty=True to be able to simulate a tty session.

        # NOTE: some images like python need to be launched with custom
        # parameters to keep stdin open. For example python image needs to be
        # started with the command `python3 -i`

        # remove local variables from kwargs
        for arg in kwargs.keys():
            if arg in locals():
                del kwargs[arg]


        kwargs = {**self._params, **kwargs}
        if 'default_command' in kwargs:
            kwargs['command'] = shlex.join(kwargs['default_command'])
            del kwargs['default_command']

        # kwargs.pop('default_command', None)
        # kwargs['command'] = cmd

        # print(f"kwargs: {kwargs}")
        # return

        # TODO: handle both output mode for tty=True/False
        logger.debug(f"running command {kwargs['command']}")
        logger.debug(f"with params {kwargs}")
        container = self._docker_client.containers.create(**kwargs)
        container.start()

        # get underlying socket
        # important to set 'stream' or attach API does not work
        _socket = container.attach_socket(params={'stdout': 1, 'stderr': 1,
                                                  'stdin': 1, 'stream': 1})


        # input()
        with DockerSocket(_socket, timeout=timeout) as _socket:
            # flush the output buffer (if any prompt)
            output = None
            try:
                flush = _socket.recv()
                _socket.setblocking(True)
                logger.debug(f"flushed output: {flush}")
                # TEST: make sure the container is ready ? use a blocking first call
                _socket.sendall(query.encode('utf-8'))

                #NOTE: delay ensures that the command is executed after the input is sent
                sleep(delay) #this should be available as a parameter

                # read the output
                output = _socket.recv()
            except socket.timeout:
                return "ERROR: timeout"


        container.kill()
        container.remove()


        # output is stored in a list of tuples (stream_type, payload)
        df = pd.DataFrame(output, columns=['stream_type', 'payload'])
        df['payload'] = df['payload'].apply(lambda x: x.decode('utf-8')).apply(lambda x: x.strip())
        df['stream_type'] = df['stream_type'].apply(lambda x: 'stdout' if x == 1 else 'stderr')
        payload = df.groupby('stream_type')['payload'].apply(''.join).to_dict()
        logger.debug(f"payload: {payload}")

        #NOTE: stderr might just contain the prompt
        if 'stdout' in payload and 'stderr' in payload and with_stderr:
            return f"STDOUT:\n {payload['stdout']}\nSTDERR:\n {payload['stderr']}"
        elif 'stderr' in payload and not 'stdout' in payload:
            return f"STDERR: {payload['stderr']}"
        else:
            return payload['stdout']
