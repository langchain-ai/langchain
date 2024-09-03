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

# Install Poetry outside of the v`irtual environment to avoid conflicts
RUN python3 -m pip install --user pipx && \
    python3 -m pipx ensurepath && \
    pipx install poetry==${POETRY_VERSION}

# Create a Python virtual environment for the project
RUN python3 -m venv ${PYTHON_VIRTUALENV_HOME} && \
    $PYTHON_VIRTUALENV_HOME/bin/pip install --upgrade pip

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
COPY libs/langchain/pyproject.toml libs/langchain/poetry.toml libs/langchain/poetry.lock ./

# Copy the langchain library for installation
COPY libs/langchain/ libs/langchain/

# Copy the core library for installation
COPY libs/core ../core

# Copy the community library for installation
COPY libs/community/ ../community/

# Copy the text-splitters library for installation
COPY libs/text-splitters/ ../text-splitters/

# Copy the partners library for installation
COPY libs/partners ../partners/

# Copy the standard-tests library for installation
COPY libs/standard-tests ../standard-tests/

# Install the Poetry dependencies (this layer will be cached as long as the dependencies don't change)
RUN poetry install --no-interaction --no-ansi --with dev,test,docs
