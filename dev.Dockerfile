# This is a Dockerfile for the Development Container

# Use the Python base image
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/devcontainers/python:0-${VARIANT} AS langchain-dev-base

USER vscode

# Define the version of Poetry to install (default is 1.4.2)
# Define the directory of python virtual environment
ARG PYTHON_VIRTUALENV_HOME=/home/vscode/langchain-py-env \
    POETRY_VERSION=1.3.2

ENV POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=true 

# Create a Python virtual environment for Poetry and install it
RUN python3 -m venv ${PYTHON_VIRTUALENV_HOME} && \
    $PYTHON_VIRTUALENV_HOME/bin/pip install --upgrade pip && \
    $PYTHON_VIRTUALENV_HOME/bin/pip install poetry==${POETRY_VERSION}

ENV PATH="$PYTHON_VIRTUALENV_HOME/bin:$PATH" \
    VIRTUAL_ENV=$PYTHON_VIRTUALENV_HOME

# Setup for bash
RUN poetry completions bash >> /home/vscode/.bash_completion && \
    echo "export PATH=$PYTHON_VIRTUALENV_HOME/bin:$PATH" >> ~/.bashrc

# Set the working directory for the app
WORKDIR /workspaces/langchain

# Use a multi-stage build to install dependencies
FROM langchain-dev-base AS langchain-dev-dependencies

ARG PYTHON_VIRTUALENV_HOME

# Copy only the dependency files for installation
COPY pyproject.toml poetry.toml ./

# Install the Poetry dependencies (this layer will be cached as long as the dependencies don't change)
RUN poetry install --no-interaction --no-ansi --with dev,test,docs