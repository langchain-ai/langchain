# TODO!: using pexpect to with containers
# TODO: add default expect pattern to image template
# TODO: pass max reads parameters for read trials
# NOTE: spawning with tty true or not gives slightly different stdout format
# NOTE: echo=False works when tty is disabled and only stdin is connected

import shlex
import os
import io
import tarfile
import time
import pandas as pd  # type: ignore
import docker
import socket

from typing import Any, Dict, Optional, Union, Type
from pydantic import BaseModel, Extra, root_validator, Field
from docker.errors import APIError, ContainerError  # type: ignore

from .images import Shell, BaseImage, get_image_template
from . import gvisor_runtime_available
from .socket_io import DockerSocket

import logging
logger = logging.getLogger(__name__)

_default_params = {
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
    """Executes arbitrary commands or payloads on containers and returns the output.

    Args:
        image (str | Type[BaseImage]): Docker image to use for execution. The
        image can be a string or a subclass of images.BaseImage.
        default_command (List[str]): Default command to use when creating the container.
    """

    _docker_client: docker.DockerClient = None  # type: ignore
    _params: Dict = Field(default_factory=Shell().dict(), skip=True)
    image: Union[str, Type[BaseImage]] = Field(default_factory=Shell, skip=True)
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
            if gvisor_runtime_available(docker.from_env()):
                self._params['runtime'] = 'runsc'

        # if not isinstance(self.image, str) and issubclass(self.image, BaseImage):
        #     self._params = {**self._params, **self.image().dict()}
        #
        # # if the user defined a custom image not pre registerd already we should
        # # not use the custom command
        # elif isinstance(self.image, str):
        #     self._params = {**_default_params(), **{'image': self.image}}

    @property
    def client(self) -> docker.DockerClient:  # type: ignore
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
                values["_params"] = {**_default_params, **{'image': image}}
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

    def _clean_kwargs(self, kwargs: dict) -> dict:
        kwargs.pop('default_command', None)
        kwargs.pop('stdin_command', None)
        return kwargs



    #FIX: default shell command should be different in run vs exec mode
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
        self._clean_kwargs(kwargs)

        args['command'] = cmd
        # print(f"args: {args}")
        # print(f"kwargs: {kwargs}")
        # return
        logger.debug(f"running command {args['command']}")
        logger.debug(f"with params {kwargs}")
        try:
            result = self._docker_client.containers.run(*(args.values()),
                                                        remove=True,
                                                        **kwargs)
            return result.decode('utf-8').strip()
        except ContainerError as e:
            return f"STDERR: {e}"

        # TODO: handle docker APIError ?
        except APIError as e:
            logger.debug(f"APIError: {e}")
            return "ERROR"

    def _flush_prompt(self, _socket):
        flush = _socket.recv()
        _socket.setblocking(True)
        logger.debug(f"flushed output: {flush}")

    def _massage_output_streams(self, output):
        df = pd.DataFrame(output, columns=['stream_type', 'payload'])
        df['payload'] = df['payload'].apply(lambda x: x.decode('utf-8'))
        df['stream_type'] = df['stream_type'].apply(
                lambda x: 'stdout' if x == 1 else 'stderr')
        payload = df.groupby('stream_type')['payload'].apply(''.join).to_dict()
        logger.debug(f"payload: {payload}")
        return payload


    # TODO: document dif between run and exec_run
    def exec_run(self, query: str, timeout: int = 5,
                 delay: float = 0.5,
                 with_stderr: bool = False,
                 flush_prompt: bool = False,
                 **kwargs: Any) -> str:
        """Run a shell command inside an ephemeral container.

        This will create a container, run the command, and then remove the
        container. the input is sent to the container's stdin through a socket
        using Docker API. It effectively simulates a tty session.

        Args:
            query (str): The command to execute.
            timeout (int): The timeout for receiving from the attached stdin.
            delay (float): The delay in seconds before running the command.
            with_stderr (bool): If True, the stderr will be included in the output
            flush_prompt (bool): If True, the prompt will be flushed before running the command.
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
        kwargs = self._clean_kwargs(kwargs)

        # exec_run requires flags for stdin so we use `stdin_command` as
        # a default command for creating the container 
        if 'stdin_command' in kwargs:
            assert isinstance(kwargs['stdin_command'], list)
            kwargs['command'] = shlex.join(kwargs['stdin_command'])
            del kwargs['stdin_command']

        # kwargs.pop('default_command', None)
        # kwargs['command'] = cmd

        # print(f"kwargs: {kwargs}")
        # return

        # TODO: handle both output mode for tty=True/False
        logger.debug(f"creating container with params {kwargs}")

        container = self._docker_client.containers.create(**kwargs)
        container.start()

        # get underlying socket
        # important to set 'stream' or attach API does not work
        _socket = container.attach_socket(params={'stdout': 1, 'stderr': 1,
                                                  'stdin': 1, 'stream': 1})


        # input()
        with DockerSocket(_socket, timeout=timeout) as _socket:
            # flush the output buffer (if any prompt)
            if flush_prompt: 
                self._flush_prompt(_socket)

            # TEST: make sure the container is ready ? use a blocking first call
            raw_input = f"{query}\n".encode('utf-8')
            _socket.sendall(raw_input)

            #NOTE: delay ensures that the command is executed after the input is sent
            time.sleep(delay) #this should be available as a parameter

            try:
                output = _socket.recv()
            except socket.timeout:
                return "ERROR: timeout"


        try:
            container.kill()
        except APIError:
            pass
        container.remove(force=True)

        if output is None:
            logger.warning("no output")
            return "ERROR"

        # output is stored in a list of tuples (stream_type, payload)
        payload = self._massage_output_streams(output)


        #NOTE: stderr might contain only the prompt
        if 'stdout' in payload and 'stderr' in payload and with_stderr:
            return f"STDOUT:\n {payload['stdout'].strip()}\nSTDERR:\n {payload['stderr']}"
        elif 'stderr' in payload and not 'stdout' in payload:
            return f"STDERR: {payload['stderr']}"
        else:
            return payload['stdout'].strip()


    def exec_attached(self, query: str, container: str,
                      delay: float = 0.5,
                      timeout: int = 5,
                      with_stderr: bool = False,
                      flush_prompt: bool = False,
                      **kwargs: Any) -> str:
        """Attach to container and exec query on it.

        This method is very similary to exec_run. It only differs in that it attaches to
        an already specifed container instead of creating a new one for each query.

        Args:
            query (str): The command to execute.
            container (str): The container to attach to.
            timeout (int): The timeout for receiving from the attached stdin.
            delay (float): The delay in seconds before running the command.
            with_stderr (bool): If True, the stderr will be included in the output
            flush_prompt (bool): If True, the prompt will be flushed before running the command.
            **kwargs: Pass extra parameters to DockerClient.container.exec_run.

        """

        # remove local variables from kwargs
        for arg in kwargs.keys():
            if arg in locals():
                del kwargs[arg]


        kwargs = {**self._params, **kwargs}
        kwargs = self._clean_kwargs(kwargs)

        logger.debug(f"attaching to container {container} with params {kwargs}")

        try:
            _container = self._docker_client.containers.get(container)
        except Exception as e:
            logger.error(f"container {container}: {e}")
            return "ERROR"

        _socket = _container.attach_socket(params={'stdout': 1, 'stderr': 1,
                                                  'stdin': 1, 'stream': 1})


        with DockerSocket(_socket, timeout=timeout) as _socket:
            # flush the output buffer (if any prompt)
            if flush_prompt: 
                self._flush_prompt(_socket)

            raw_input = f"{query}\n".encode('utf-8')
            _socket.sendall(raw_input)

            #NOTE: delay ensures that the command is executed after the input is sent
            time.sleep(delay) #this should be available as a parameter

            try:
                output = _socket.recv()
            except socket.timeout:
                return "ERROR: timeout"

        if output is None:
            logger.warning("no output")
            return "ERROR"

        payload = self._massage_output_streams(output)
        print(payload)

        #NOTE: stderr might contain only the prompt
        if 'stdout' in payload and 'stderr' in payload and with_stderr:
            return f"STDOUT:\n {payload['stdout'].strip()}\nSTDERR:\n {payload['stderr']}"
        elif 'stderr' in payload and not 'stdout' in payload:
            return f"STDERR: {payload['stderr']}"
        else:
            return payload['stdout'].strip()



    #WIP method that will copy the given payload to the container filesystem then
    # invoke the command on the file and return the output
    def run_file(self, payload: bytes, filename: Optional[str] = None,
                 **kwargs: Any) -> str:
        """Run arbitrary shell command inside an ephemeral container on the
        specified input payload."""


        for arg in kwargs.keys():
            if arg in locals():
                del kwargs[arg]

        kwargs = {**self._params, **kwargs}
        self._clean_kwargs(kwargs)

        kwargs['command'] = '/bin/sh'

        k_file_location = '/tmp/payload'
        if filename is not None:
            # store at /tmp/file_name
            # strip all leading path components
            file_loc = os.path.basename(filename)
            k_file_location = f'/tmp/{file_loc}'

        # print(kwargs)
        # return

        # create a container with the given payload
        # container = self._docker_client.containers.create(**kwargs)
        # container.start()
        container = self._docker_client.containers.list()[0]
        print(container.short_id)


        # copy the payload to the container
        try:
            # put the data in tar archive at the path specified by k_file_location
            archive = io.BytesIO()
            with tarfile.TarFile(fileobj=archive, mode='w') as tar:
                tarinfo = tarfile.TarInfo(name='test-archive')
                tarinfo.size = len(payload)
                tarinfo.mtime = int(time.time())
                tar.addfile(tarinfo, io.BytesIO(payload))
            archive.seek(0)

            # store archive on local host at /tmp/test
            # with open('/tmp/test', 'wb') as f:
            #     f.write(archive.read())


            container.put_archive(path='/', data=archive)
        except APIError as e:
            logger.error(f"Error: {e}")
            return "ERROR"

        #execute the command
        exit_code, out = container.exec_run(['sh', k_file_location])
        print(f"exit_code: {exit_code}")
        print(f"out: {out}")


        # try:
        #     container.kill()
        # except APIError:
        #     pass
        # container.remove(force=True)

        return ""
