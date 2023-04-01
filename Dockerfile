# Use the Python base image
FROM python:3.11.2-bullseye AS builder

# Print Python version
RUN echo "Python version:" && python --version && echo ""

# Install Poetry
RUN echo "Installing Poetry..." && \
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -

# Add Poetry to PATH
ENV PATH="${PATH}:/root/.local/bin"

# Test if Poetry is added to PATH
RUN echo "Poetry version:" && poetry --version && echo ""

# Set working directory
WORKDIR /app

# Use a multi-stage build to install dependencies
FROM builder AS dependencies

# Copy only the dependency files for installation
COPY pyproject.toml poetry.lock poetry.toml ./

# Install Poetry dependencies (this layer will be cached as long as the dependencies don't change)
RUN poetry install --no-interaction --no-ansi

# Use a multi-stage build to run tests
FROM dependencies AS tests

# Copy the rest of the app source code (this layer will be invalidated and rebuilt whenever the source code changes)
COPY . .

# Set entrypoint to run tests
ENTRYPOINT ["poetry", "run", "pytest"]

# Set default command to run all unit tests
CMD ["tests/unit_tests"]
