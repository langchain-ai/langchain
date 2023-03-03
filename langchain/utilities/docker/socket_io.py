"""Low level socket IO for docker API."""
import struct
import logging
from typing import Any

logger = logging.getLogger(__name__)

SOCK_BUF_SIZE = 1024

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
        # SIZE1, SIZE2, SIZE3, SIZE4 are the four bytes of the uint32 size encoded as
        # big endian.
        #
        # Following the header is the payload, which is the specified number of bytes of
        # STREAM_TYPE.
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
