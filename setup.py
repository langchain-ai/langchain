"""Set up the package."""
from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="langchain",
    version_format="0.0.0",
    packages=find_packages(),
    description="Building LLM empowered applications",
    install_requires=["pydantic"],
    long_description=long_description,
    license="MIT",
    url="https://github.com/hwchase17/langchain",
)
