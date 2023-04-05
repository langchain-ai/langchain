## Integrations Tests

To run the tests, you will need to install the following software:

- Docker
- Python 3.8 or later

This repository contains functional tests for several search engines and databases. The tests aim to verify the correct
behavior of the engines and databases according to their specifications and requirements.

We have optional group `test_integration` in the `pyproject.toml` file. This group should contain dependencies for
integration tests and can be installed using the command:

```bash
poetry install --with test_integration
```

Any new dependencies should be added by running:

```bash
poetry add some_new_deps --group "test_integration" 
```

Before running any tests, you should start a specific Docker container that has all the necessary dependencies
installed. For instance, we use the `elasticsearch.yml` container for `test_elasticsearch.py`:

```bash
cd tests/integration_tests/vectorstores/docker-compose
docker-compose -f elasticsearch.yml up
```