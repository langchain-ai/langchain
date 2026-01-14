(Replace this entire block of text)

Read the full contributing guidelines: https://docs.langchain.com/oss/python/contributing/overview

Thank you for contributing to LangChain! Follow these steps to have your pull request considered as ready for review.

1. PR title: Should follow the format: TYPE(SCOPE): DESCRIPTION

  - Examples:
    - fix(anthropic): resolve flag parsing error
    - feat(core): add multi-tenant support
    - test(openai): update API usage tests
  - Allowed TYPE and SCOPE values: https://github.com/langchain-ai/langchain/blob/master/.github/workflows/pr_lint.yml#L15-L33

2. PR description:

  - Write 1-2 sentences summarizing the change.
  - If this PR addresses a specific issue, please include "Fixes #ISSUE_NUMBER" in the description to automatically close the issue when the PR is merged.
  - If there are any breaking changes, please clearly describe them.
  - If this PR depends on another PR being merged first, please include "Depends on #PR_NUMBER" inthe description.

3. Run `make format`, `make lint` and `make test` from the root of the package(s) you've modified.

  - We will not consider a PR unless these three are passing in CI.

Additional guidelines:

  - We ask that if you use generative AI for your contribution, you include a disclaimer.
  - PRs should not touch more than one package unless absolutely necessary.
  - Do not update the `uv.lock` files unless or add dependencies to `pyproject.toml` files (even optional ones) unless you have explicit permission to do so by a maintainer.
