(Replace this entire block of text)

Thank you for contributing to LangChain! Follow these steps to mark your pull request as ready for review. **If any of these steps are not completed, your PR will not be considered for review.**

- [ ] **PR title**: Follows the format: {TYPE}({SCOPE}): {DESCRIPTION}
  - Examples:
    - feat(core): add multi-tenant support
    - fix(cli): resolve flag parsing error
    - docs(openai): update API usage examples
  - Allowed `{TYPE}` values:
    - feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert, release
  - Allowed `{SCOPE}` values (optional):
    - core, cli, langchain, standard-tests, text-splitters, docs, anthropic, chroma, deepseek, exa, fireworks, groq, huggingface, mistralai, nomic, ollama, openai, perplexity, prompty, qdrant, xai, infra
  - Once you've written the title, please delete this checklist item; do not include it in the PR.

- [ ] **PR message**: ***Delete this entire checklist*** and replace with
  - **Description:** a description of the change. Include a [closing keyword](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword) if applicable to a relevant issue.
  - **Issue:** the issue # it fixes, if applicable (e.g. Fixes #123)
  - **Dependencies:** any dependencies required for this change

- [ ] **Lint and test**: Run `make format`, `make lint` and `make test` from the root of the package(s) you've modified. **We will not consider a PR unless these three are passing in CI.** See [contribution guidelines](https://python.langchain.com/docs/contributing/) for more.

Additional guidelines:

- Most PRs should not touch more than one package.
- Please do not add dependencies to `pyproject.toml` files (even optional ones) unless they are **required** for unit tests. Likewise, please do not update the `uv.lock` files unless you are adding a required dependency.
- Changes should be backwards compatible.
- Make sure optional dependencies are imported within a function.
