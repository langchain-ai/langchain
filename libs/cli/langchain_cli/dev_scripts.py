"""
Development Scripts for template packages
"""

from fastapi import FastAPI
from langserve import add_routes
from langserve.packages import get_langserve_export

from langchain_cli.utils.packages import get_package_root


def create_demo_server():
    """
    Creates a demo server for the current template.
    """
    app = FastAPI()
    package_root = get_package_root()
    pyproject = package_root / "pyproject.toml"
    try:
        package = get_langserve_export(pyproject)

        mod = __import__(package["module"], fromlist=[package["attr"]])

        chain = getattr(mod, package["attr"])
        add_routes(app, chain)
    except KeyError as e:
        raise KeyError("Missing fields from pyproject.toml") from e
    except ImportError as e:
        raise ImportError("Could not import module defined in pyproject.toml") from e

    return app
