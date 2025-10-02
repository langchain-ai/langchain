# langchain-groq

## Welcome to Groq! 🚀

At Groq, we've developed the world's first Language Processing Unit™, or LPU. The Groq LPU has a deterministic, single core streaming architecture that sets the standard for GenAI inference speed with predictable and repeatable performance for any given workload.

Beyond the architecture, our software is designed to empower developers like you with the tools you need to create innovative, powerful AI applications. With Groq as your engine, you can:

* Achieve uncompromised low latency and performance for real-time AI and HPC inferences 🔥
* Know the exact performance and compute time for any given workload 🔮
* Take advantage of our cutting-edge technology to stay ahead of the competition 💪

Want more Groq? Check out our [website](https://groq.com) for more resources and join our [Discord community](https://discord.gg/JvNsBDKeCG) to connect with our developers!

## Installation and Setup

Install the integration package:

```bash
pip install langchain-groq
```

Request an [API key](https://console.groq.com/login?utm_source=langchain&utm_content=package_readme) and set it as an environment variable

```bash
export GROQ_API_KEY=gsk_...
```

## Chat Model

See a [usage example](https://python.langchain.com/docs/integrations/chat/groq).

## Development

To develop the `langchain-groq` package, you'll need to follow these instructions:

### Install dev dependencies

```bash
uv sync --group lint --group test
```

### Build the package

```bash
uv build
```

### Run unit tests

Unit tests live in `tests/unit_tests` and SHOULD NOT require an internet connection or a valid API KEY.  Run unit tests with

```bash
make tests
```

### Run integration tests

Integration tests live in `tests/integration_tests` and require a connection to the Groq API and a valid API KEY.

```bash
make integration_tests
```

### Lint & Format

Run additional tests and linters to ensure your code is up to standard.

```bash
make lint check_imports
```
