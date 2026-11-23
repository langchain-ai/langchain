"""Run the two guarded MCP servers the graph factory talks to.

`graph_factory.py` points `SERVERS` at fixed ports; this starts a server on
each. Both verify tokens against the shared keypair (`MCP_DEMO_KEYFILE`), the
same one `langgraph dev` mints with, so a token minted in the factory process
validates here.

    MCP_DEMO_KEYFILE=/path/to/keys.json uv run examples/mcp/_fleet_servers.py
"""

from __future__ import annotations

import multiprocessing

from _servers import PUBLIC_KEY, run_guarded_server

PORTS = {"calendar": 8001, "docs": 8002}


def main() -> None:
    procs = [
        multiprocessing.Process(
            target=run_guarded_server,
            args=("127.0.0.1", port, name, PUBLIC_KEY),
            daemon=True,
        )
        for name, port in PORTS.items()
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
