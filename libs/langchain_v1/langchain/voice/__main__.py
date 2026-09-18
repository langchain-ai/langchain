"""Run an importable VoiceAgent from the command line."""

from __future__ import annotations

import argparse
import asyncio
import importlib
from typing import Any

from langchain.voice.agent import VoiceAgent
from langchain.voice.transports.websocket import WebSocketServer


def _load_agent(reference: str) -> VoiceAgent:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        msg = "agent must use the form module:attribute"
        raise ValueError(msg)
    value: Any = getattr(importlib.import_module(module_name), attribute)
    agent = value() if callable(value) and not isinstance(value, VoiceAgent) else value
    if not isinstance(agent, VoiceAgent):
        msg = f"{reference} did not resolve to a VoiceAgent"
        raise TypeError(msg)
    return agent


def main() -> None:
    """Serve an imported voice agent over WebSocket."""
    parser = argparse.ArgumentParser(description="Serve a LangChain Voice agent over WebSocket")
    parser.add_argument("agent", help="VoiceAgent import path, such as app:agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    agent = _load_agent(args.agent)
    asyncio.run(WebSocketServer(agent).serve_forever(args.host, args.port))


if __name__ == "__main__":
    main()
