# Cheat sheet

Build with the base dev image plus my snowflake dependencies

```
docker-compose -f docker-compose.yaml -f compose.local.yaml build
```

Start container using azure LLM or openai LLM and the run test

## Azure LLM

```
docker-compose -f docker-compose.yaml -f compose.local.yaml -f compose.azure.yaml up
docker exec -it langchain_azure pytest --pdb -s ../tests/integration_tests/chains/test_cpal.py
```

## OpenAI LLM for Jupyter notebook development

```
docker-compose -f docker-compose.yaml -f compose.local.yaml -f compose.openai.yaml up
docker exec -it langchain_openai pytest --pdb -s ../tests/integration_tests/chains/test_cpal.py
```
