"""
Development Scripts for template packages
"""

from fastapi import FastAPI
from langserve.packages import add_package_route

from langchain_cli.utils.packages import get_package_root


def create_demo_server():
    """
    Creates a demo server for the current template.
    """
    app = FastAPI()
    package_root = get_package_root()
    add_package_route(app, package_root, "")
    return app
