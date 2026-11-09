"""Drive the graph factory through a real `langgraph dev` server with auth.

This is the end-to-end check the example describes but cannot show inline: two
users hit one deployment, each authenticates with their own `x-api-key`, and
each run's agent talks to the MCP fleet as that user. It:

  1. mints one shared keypair (so the token the factory mints validates on the
     servers) into a temp `MCP_DEMO_KEYFILE`,
  2. starts the two guarded MCP servers on the ports `graph_factory.SERVERS`
     names,
  3. starts `langgraph dev` over `langgraph.json` (which wires `auth.py`),
  4. runs the `fleet` graph once per user and checks each agent reports its own
     identity from the servers' `whoami` tools.

    ANTHROPIC_API_KEY=... uv run examples/mcp/run_graph_factory_demo.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fastmcp.server.auth.providers.jwt import RSAKeyPair
from langgraph_sdk import get_client

HERE = Path(__file__).parent
DEV_URL = "http://127.0.0.1:2024"
USERS = ("alice", "bob")


def _write_keyfile() -> Path:
    """Serialize one keypair every process reads, so tokens validate across them."""
    keys = RSAKeyPair.generate()
    path = Path(tempfile.mkstemp(suffix=".json", prefix="mcp-demo-keys-")[1])
    path.write_text(
        json.dumps(
            {"private_key": keys.private_key.get_secret_value(), "public_key": keys.public_key}
        )
    )
    return path


async def _wait_until_up(url: str, timeout: float = 60.0) -> None:
    """Poll until the dev server has registered the `fleet` graph, or time out."""
    deadline = time.monotonic() + timeout
    client = get_client(url=url, headers={"x-user-id": "healthcheck"})
    while time.monotonic() < deadline:
        with contextlib.suppress(Exception):
            assistants = await client.assistants.search(graph_id="fleet")
            if assistants:
                return
        await asyncio.sleep(1.0)
    msg = f"dev server at {url} never registered the 'fleet' graph"
    raise RuntimeError(msg)


async def _run_as(user: str) -> str:
    """Run the fleet graph as `user` and return the agent's final text."""
    # The `x-user-id` header is what `auth.py` turns into this run's identity.
    client = get_client(url=DEV_URL, headers={"x-user-id": user})
    thread = await client.threads.create()
    result = await client.runs.wait(
        thread["thread_id"],
        "fleet",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Call both whoami tools. Reply with exactly: calendar=<value> docs=<value>."
                    ),
                }
            ]
        },
    )
    return result["messages"][-1]["content"]


async def _drive() -> int:
    await _wait_until_up(DEV_URL)
    failures = 0
    for user in USERS:
        reply = await _run_as(user)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        ok = f"calendar={user}" in text and f"docs={user}" in text
        print(f"{user:6} -> {'PASS' if ok else 'FAIL'}: {text!r}")
        failures += not ok
    return failures


def main() -> int:
    """Start the fleet + dev server, run every user through it, and report."""
    keyfile = _write_keyfile()
    env = {**os.environ, "MCP_DEMO_KEYFILE": str(keyfile)}
    # Use the `langgraph` next to this interpreter, not whatever is first on
    # PATH: the graph imports this venv's deps (`httpx2`, `fastmcp`, ...), so it
    # has to run under this venv, not a globally installed CLI.
    langgraph = Path(sys.executable).with_name("langgraph")
    if not langgraph.exists():
        msg = f"`langgraph` CLI not found at {langgraph}; install langgraph-cli[inmem]."
        raise RuntimeError(msg)

    # `langgraph dev` persists its graph registry under `.langgraph_api`; a
    # stale one from an earlier run would shadow this config's `fleet` graph.
    shutil.rmtree(HERE / ".langgraph_api", ignore_errors=True)

    servers = subprocess.Popen(  # noqa: S603
        [sys.executable, str(HERE / "_fleet_servers.py")], env=env, cwd=HERE
    )
    devlog = Path("/tmp/lg-driver.log")  # noqa: S108
    print(f"(dev log -> {devlog})")
    dev = subprocess.Popen(  # noqa: S603
        [
            str(langgraph),
            "dev",
            "--no-browser",
            "--port",
            "2024",
            "--config",
            "langgraph.json",
            # A transitive dependency (`jsonschema_specifications`) scans its
            # package data synchronously the first time a tool schema is
            # validated. `langgraph dev` rejects blocking calls on the event
            # loop by default; allow them here so the demo can run.
            "--allow-blocking",
        ],
        env=env,
        cwd=HERE,
        stdout=devlog.open("w"),
        stderr=subprocess.STDOUT,
    )
    try:
        return asyncio.run(_drive())
    finally:
        for proc in (dev, servers):
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=10)
        keyfile.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
