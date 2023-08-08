# This is a Dockerfile for running unit tests

ARG POETRY_HOME=/opt/poetry

# Use the Python base image
FROM python:3.11.2-bullseye AS builder

# Define the version of Poetry to install (default is 1.4.2)
ARG POETRY_VERSION=1.4.2

# Define the directory to install Poetry to (default is /opt/poetry)
ARG POETRY_HOME

# Create a Python virtual environment for Poetry and install it
RUN python3 -m venv ${POETRY_HOME} && \
    $POETRY_HOME/bin/pip install --upgrade pip && \
    $POETRY_HOME/bin/pip install poetry==${POETRY_VERSION}

# Test if Poetry is installed in the expected path
RUN echo "Poetry version:" && $POETRY_HOME/bin/poetry --version

# Set the working directory for the app
WORKDIR /app

# Use a multi-stage build to install dependencies
FROM builder AS dependencies

ARG POETRY_HOME

# Copy only the dependency files for installation
COPY pyproject.toml poetry.lock poetry.toml ./

# Install the Poetry dependencies (this layer will be cached as long as the dependencies don't change)
RUN $POETRY_HOME/bin/poetry install --no-interaction --no-ansi --with test

# Use a multi-stage build to run tests
FROM dependencies AS tests

# Copy the rest of the app source code (this layer will be invalidated and rebuilt whenever the source code changes)
COPY . .

RUN /opt/poetry/bin/poetry install --no-interaction --no-ansi --with test

# Set the entrypoint to run tests using Poetry
ENTRYPOINT ["/opt/poetry/bin/poetry", "run", "pytest"]

# Set the default command to run all unit tests
CMD ["tests/unit_tests"]
