Thank you for contributing to LangChain!

- [ ] **PR title**: Follows the [Conventional Commits specification](https://www.conventionalcommits.org/en/v1.0.0/): {TYPE}({SCOPE}): {DESCRIPTION}
    - The `{DESCRIPTION}` must not start with an uppercase letter.
    - Examples:
        - feat(core): add multi-tenant support
        - fix(cli): resolve flag parsing error
        - docs(openai): update API usage examples
    - Allowed `{TYPE}` values:
        - feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert, release
    - Allowed `{SCOPE}` values (optional):
        - core, cli, langchain, standard-tests, docs, anthropic, chroma, deepseek, exa, fireworks, groq, huggingface, mistralai, nomic, ollama, openai, perplexity, prompty, qdrant, xai

- [ ] **PR message**: ***Delete this entire checklist*** and replace with
    - **Description:** a description of the change. Include a [closing keyword](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword) if applicable.
    - **Issue:** the issue # it fixes, if applicable
    - **Dependencies:** any dependencies required for this change
    - **Twitter handle:** if your PR gets announced, and you'd like a mention, we'll gladly shout you out!


- [ ] **Add tests and docs**: If you're adding a new integration, please include
  1. A test for the integration, preferably unit tests that do not rely on network access,
  2. An example notebook showing its use. It lives in `docs/docs/integrations` directory.


- [ ] **Lint and test**: Run `make format`, `make lint` and `make test` from the root of the package(s) you've modified. We will not consider a PR unless these three are passing in CI. See contribution guidelines for more: https://python.langchain.com/docs/contributing/

Additional guidelines:
- Make sure optional dependencies are imported within a function.
- Please do not add dependencies to `pyproject.toml` files (even optional ones) unless they are **required** for unit tests.
- Most PRs should not touch more than one package.
- Changes should be backwards compatible.

If no one reviews your PR within a few days, please @-mention one of eyurtsev, ccurme, mdrxy.
