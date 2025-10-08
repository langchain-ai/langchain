# langchain-ollama

This package contains the LangChain integration with Ollama

View the [documentation](https://docs.langchain.com/oss/python/integrations/providers/ollama) for more details.

## Development

### Running Tests

To run integration tests (`make integration_tests`), you will need the following models installed in your Ollama server:

- `llama3.1`
- `deepseek-r1:1.5b`
- `gpt-oss:20b`

Install these models by running:

```bash
ollama pull <name-of-model>
```
