# Readme tests(draft)

## Integrations Tests

### Prepare

This repository contains functional tests for several search engines and databases. The
tests aim to verify the correct behavior of the engines and databases according to their
specifications and requirements.

To run some integration tests, such as tests located in
`tests/integration_tests/vectorstores/`, you will need to install the following
software:

- Docker
- Python 3.8.1 or later

We have optional group `test_integration` in the `pyproject.toml` file. This group
should contain dependencies for the integration tests and can be installed using the
command:

```bash
poetry install --with test_integration
```

Any new dependencies should be added by running:

```bash
# add package and install it after adding:
poetry add tiktoken@latest --group "test_integration" && poetry install --with test_integration
```

Before running any tests, you should start a specific Docker container that has all the
necessary dependencies installed. For instance, we use the `elasticsearch.yml` container
for `test_elasticsearch.py`:

```bash
cd tests/integration_tests/vectorstores/docker-compose
docker-compose -f elasticsearch.yml up
```

### Prepare environment variables for local testing:

- copy `tests/.env.example` to `tests/.env`
- set variables in `tests/.env` file, e.g `OPENAI_API_KEY`

Additionally, it's important to note that some integration tests may require certain
environment variables to be set, such as `OPENAI_API_KEY`. Be sure to set any required
environment variables before running the tests to ensure they run correctly.

### Recording HTTP interactions with pytest-vcr

Some of the integration tests in this repository involve making HTTP requests to
external services. To prevent these requests from being made every time the tests are
run, we use pytest-vcr to record and replay HTTP interactions.

When running tests in a CI/CD pipeline, you may not want to modify the existing
cassettes. You can use the --vcr-record=none command-line option to disable recording
new cassettes. Here's an example:

```bash
pytest --log-cli-level=10 tests/integration_tests/vectorstores/test_pinecone.py --vcr-record=none
pytest tests/integration_tests/vectorstores/test_elasticsearch.py --vcr-record=none

```

### Run some tests with coverage:

```bash
pytest tests/integration_tests/vectorstores/test_elasticsearch.py --cov=langchain --cov-report=html
start "" htmlcov/index.html || open htmlcov/index.html

```