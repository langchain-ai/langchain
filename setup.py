"""Set up the package."""
from pathlib import Path

from setuptools import find_packages, setup

with open(Path(__file__).absolute().parents[0] / "langchain" / "VERSION") as _f:
    __version__ = _f.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="langchain",
    version=__version__,
    packages=find_packages(),
    description="Building applications with LLMs through composability",
    install_requires=["pydantic", "sqlalchemy", "numpy"],
    long_description=long_description,
    license="MIT",
    url="https://github.com/hwchase17/langchain",
    include_package_data=True,
    long_description_content_type="text/markdown",
)
