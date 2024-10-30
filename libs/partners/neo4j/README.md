# 🦜️🔗 Langchain Neo4j

This package contains the LangChain integration with Neo4j

## 📦 Installation

```bash
pip install -U langchain-neo4j
```

## 🧪 Tests

Install the test dependencies to run the tests:

```bash
poetry install --with test,test_integration
```

### Unit Tests

Run the unit tests using:

```bash
make tests
```

### Integration Tests

1. Export your OpenAI API key:

    ```bash
    export OPENAI_API_KEY=sk-...
    ```

2. Start the Neo4j instance using Docker:

    ```bash
    cd tests/integration_tests/docker-compose
    docker-compose -f neo4j.yml up
    ```

3. Run the tests:

    ```bash
    make integration_tests
    ```
