<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-dark.svg">
    <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-light.svg">
    <img alt="LangChain Logo" src=".github/images/logo-dark.svg" width="80%">
  </picture>
</p>

<p align="center">
    The platform for reliable agents.
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT" target="_blank">
      <img src="https://img.shields.io/pypi/l/langchain" alt="PyPI - License">
  </a>
  <a href="https://pypistats.org/packages/langchain" target="_blank">
      <img src="https://img.shields.io/pepy/dt/langchain" alt="PyPI - Downloads">
  </a>
  <a href="https://pypi.org/project/langchain/#history" target="_blank">
      <img src="https://img.shields.io/pypi/v/langchain?label=%20" alt="Version">
  </a>
  <a href="https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/langchain-ai/langchain" target="_blank">
      <img src="https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode" alt="Open in Dev Containers">
  </a>
  <a href="https://codespaces.new/langchain-ai/langchain" target="_blank">
      <img src="https://github.com/codespaces/badge.svg" alt="Open in Github Codespace" title="Open in Github Codespace" width="150" height="20">
  </a>
  <a href="https://codspeed.io/langchain-ai/langchain" target="_blank">
      <img src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" alt="CodSpeed Badge">
  </a>
  <a href="https://twitter.com/langchainai" target="_blank">
      <img src="https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI" alt="Twitter / X">
  </a>
</p>

LangChain is a framework for building agents and LLM-powered applications. It helps you chain together interoperable components and third-party integrations to simplify AI application development —  all while future-proofing decisions as the underlying technology evolves.

```bash
pip install langchain
```

If you're looking for more advanced customization or agent orchestration, check out [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), our framework for building controllable agent workflows.

---

**Documentation**: To learn more about LangChain, check out [the docs](https://docs.langchain.com/oss/python/langchain/overview).

**Discussions**: Visit the [LangChain Forum](https://forum.langchain.com) to connect with the community and share all of your technical questions, ideas, and feedback.

> [!NOTE]
> Looking for the JS/TS library? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Why use LangChain?

LangChain helps developers build applications powered by LLMs through a standard interface for models, embeddings, vector stores, and more.

Use LangChain for:

- **Real-time data augmentation**. Easily connect LLMs to diverse data sources and external/internal systems, drawing from LangChain’s vast library of integrations with model providers, tools, vector stores, retrievers, and more.
- **Model interoperability**. Swap models in and out as your engineering team experiments to find the best choice for your application’s needs. As the industry frontier evolves, adapt quickly — LangChain’s abstractions keep you moving without losing momentum.

## LangChain ecosystem

While the LangChain framework can be used standalone, it also integrates seamlessly with any LangChain product, giving developers a full suite of tools when building LLM applications.

To improve your LLM application development, pair LangChain with:

- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) - Build agents that can reliably handle complex tasks with LangGraph, our low-level agent orchestration framework. LangGraph offers customizable architecture, long-term memory, and human-in-the-loop workflows — and is trusted in production by companies like LinkedIn, Uber, Klarna, and GitLab.
- [LangSmith](https://www.langchain.com/langsmith) - Helpful for agent evals and observability. Debug poor-performing LLM app runs, evaluate agent trajectories, gain visibility in production, and improve performance over time.
- [LangSmith Deployment](https://docs.langchain.com/langsmith/deployments) - Deploy and scale agents effortlessly with a purpose-built deployment platform for long-running, stateful workflows. Discover, reuse, configure, and share agents across teams — and iterate quickly with visual prototyping in [LangSmith Studio](https://docs.langchain.com/langsmith/studio).

## Additional resources

- [API Reference](https://reference.langchain.com/python): Detailed reference on navigating base packages and integrations for LangChain.
- [Integrations](https://docs.langchain.com/oss/python/integrations/providers/overview): List of LangChain integrations, including chat & embedding models, tools & toolkits, and more
- [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview): Learn how to contribute to LangChain and find good first issues.
- [Code of Conduct](https://github.com/langchain-ai/langchain/blob/master/.github/CODE_OF_CONDUCT.md): Our community guidelines and standards for participation.
