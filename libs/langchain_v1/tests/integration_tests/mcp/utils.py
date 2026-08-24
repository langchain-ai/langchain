import contextlib
import multiprocessing
import socket
import time
from collections.abc import Generator

import uvicorn
from dirty_equals import IsStr
from mcp.server.websocket import websocket_server
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute

from tests.integration_tests.mcp.servers.time_server import mcp as time_mcp

# Helper for matching auto-generated LangChain content block IDs
IsLangChainID = IsStr(regex=r"lc_.*")


def make_server_app() -> Starlette:
    server = time_mcp._mcp_server

    async def handle_ws(websocket):
        async with websocket_server(websocket.scope, websocket.receive, websocket.send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    app = Starlette(routes=[WebSocketRoute("/ws", endpoint=handle_ws)])

    return app


def run_server(server_port: int) -> None:
    app = make_server_app()
    server = uvicorn.Server(
        config=uvicorn.Config(app=app, host="127.0.0.1", port=server_port, log_level="error"),
    )
    server.run()

    # Give server time to start
    while not server.started:
        time.sleep(0.5)


def run_streamable_http_server(server_factory, server_port: int) -> None:
    """Run a FastMCP server in a separate process exposing a streamable HTTP."""
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
        raise RuntimeError(f"Server failed to start after {max_attempts} attempts")

    try:
        yield
    finally:
        # Signal the server to stop
        proc.kill()
        proc.join(timeout=2)
        if proc.is_alive():
            raise RuntimeError("Server process is still alive after attempting to terminate it")
