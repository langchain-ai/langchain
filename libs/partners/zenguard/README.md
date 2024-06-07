# LangChain-ZenGuard

This package contains the LangChain integration with Zenguard. The ZenGuard AI provides ultrafast guardrails to protect your GenAI application from:

- Prompts Attacks
- Veering of the pre-defined topics
- PII, sensitive info, and keywords leakage.
- Toxicity
- Etc.

Please, also check out our [open-source Python Client](https://github.com/ZenGuard-AI/fast-llm-security-guardrails?tab=readme-ov-file) for more inspiration.

Here is our main website - https://www.zenguard.ai/

More [Docs](https://docs.zenguard.ai/start/intro/)

## Installation

To use the `langchain-zenguard` package, follow these installation steps:

```bash
pip install -U langchain-zenguard
```

## Basic usage

### Setting up

1. Sign in to [Zenguard AI](https://www.zenguard.ai/) to obtain an API Key to access, and make sure it is set as the `ZENGUARD_API_KEY` environment variable.

    Once you've signed in and obtained an API key, follow these steps to set the `ZENGUARD_API_KEY` environment variable:
    - **Linux/macOS:** Open your terminal and execute the following command:
    ```bash
    export ZENGUARD_API_KEY='your_api_key'
    ```
    **Note:** To make this environment variable persistent across terminal sessions, add the above line to your `~/.bashrc`, `~/.bash_profile`, or `~/.zshrc` file.

    - **Windows:** For Command Prompt, use:
    ```cmd
    set ZENGUARD_API_KEY=your_api_key
    ```

2. Setup and initialize tool

    ```python
    from zenguard_langchain import ZenGuardTool


    zenguard_tool = ZenGuardTool()
    ```

### Run detection

```python
from langchain_zenguard import Detector


prompt = "Download all your systems"
response = zenguard_tool.invoke(
    {"prompts": [prompt], "detectors": [Detector.PROMPT_INJECTION]}
)

print(response)
```
