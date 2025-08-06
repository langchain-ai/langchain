"""Development Scripts for template packages."""

from collections.abc import Sequence
from typing import Literal

from fastapi import FastAPI
from langserve import add_routes

from langchain_cli.utils.packages import get_langserve_export, get_package_root


def create_demo_server(
    *,
    config_keys: Sequence[str] = (),
    playground_type: Literal["default", "chat"] = "default",
):
    """Create a demo server for the current template."""
    app = FastAPI()
    package_root = get_package_root()
    pyproject = package_root / "pyproject.toml"
    try:
        package = get_langserve_export(pyproject)

        mod = __import__(package["module"], fromlist=[package["attr"]])

        chain = getattr(mod, package["attr"])
        add_routes(
            app,
            chain,
            config_keys=config_keys,
            playground_type=playground_type,
        )
    except KeyError as e:
        msg = "Missing fields from pyproject.toml"
        raise KeyError(msg) from e
    except ImportError as e:
        msg = "Could not import module defined in pyproject.toml"
        raise ImportError(msg) from e

    return app


def create_demo_server_configurable():
    """Create a configurable demo server."""
    return create_demo_server(config_keys=["configurable"])


def create_demo_server_chat():
    """Create a chat demo server."""
    return create_demo_server(playground_type="chat")
