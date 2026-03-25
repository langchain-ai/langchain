Fixes #

<!-- Replace everything above this line with a 1-2 sentence description of your change. Keep the "Fixes #xx" keyword and update the issue number. -->

Read the full contributing guidelines: https://docs.langchain.com/oss/python/contributing/overview

> **All contributions must be in English.** See the [language policy](https://docs.langchain.com/oss/python/contributing/overview#language-policy).

If you paste a large clearly AI generated description here your PR may be IGNORED or CLOSED!

Thank you for contributing to LangChain! Follow these steps to have your pull request considered as ready for review.

1. PR title: Should follow the format: TYPE(SCOPE): DESCRIPTION

  - Examples:
    - fix(anthropic): resolve flag parsing error
    - feat(core): add multi-tenant support
    - test(openai): update API usage tests
  - Allowed TYPE and SCOPE values: https://github.com/langchain-ai/langchain/blob/master/.github/workflows/pr_lint.yml#L15-L33

2. PR description:

  - Write 1-2 sentences summarizing the change.
  - The `Fixes #xx` line at the top is **required** for external contributions — update the issue number and keep the keyword. This links your PR to the approved issue and auto-closes it on merge.
  - If there are any breaking changes, please clearly describe them.
  - If this PR depends on another PR being merged first, please include "Depends on #PR_NUMBER" in the description.

3. Run `make format`, `make lint` and `make test` from the root of the package(s) you've modified.

  - We will not consider a PR unless these three are passing in CI.

4. How did you verify your code works?

Additional guidelines:

  - All external PRs must link to an issue or discussion where a solution has been approved by a maintainer, and you must be assigned to that issue. PRs without prior approval will be closed.
  - PRs should not touch more than one package unless absolutely necessary.
  - Do not update the `uv.lock` files or add dependencies to `pyproject.toml` files (even optional ones) unless you have explicit permission to do so by a maintainer.

## Social handles (optional)
<!-- If you'd like a shoutout on release, add your socials below -->
Twitter: @
LinkedIn: https://linkedin.com/in/
