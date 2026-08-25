import contextlib
import multiprocessing
import socket
import time
from collections.abc import Generator

import uvicorn
from dirty_equals import IsStr

# Helper for matching auto-generated LangChain content block IDs
IsLangChainID = IsStr(regex=r"lc_.*")


def run_streamable_http_server(server_factory, server_port: int) -> None:
    """Run a MCPServer server in a separate process exposing a streamable HTTP."""
    server = server_factory()
    app = server.streamable_http_app()
    uvicorn_server = uvicorn.Server(
        config=uvicorn.Config(app=app, host="127.0.0.1", port=server_port, log_level="error"),
    )
    uvicorn_server.run()


@contextlib.contextmanager
def run_streamable_http(server_factory, server_port: int) -> Generator[None, None, None]:
    """Run the server in a separate process exposing a streamable HTTP endpoint.

    The endpoint will be available at `http://localhost:{server_port}/mcp`.
    """
    proc = multiprocessing.Process(
        target=run_streamable_http_server,
        kwargs={"server_factory": server_factory, "server_port": server_port},
        daemon=True,
    )
    proc.start()

    # Wait for server to be running
    max_attempts = 20
    attempt = 0

    while attempt < max_attempts:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("127.0.0.1", server_port))
                break
        except ConnectionRefusedError:
            time.sleep(0.1)
            attempt += 1
    else:
        msg = f"Server failed to start after {max_attempts} attempts"
        raise RuntimeError(msg)

    try:
        yield
    finally:
        # Signal the server to stop
        proc.kill()
        proc.join(timeout=2)
        if proc.is_alive():
            msg = "Server process is still alive after attempting to terminate it"
            raise RuntimeError(msg)
