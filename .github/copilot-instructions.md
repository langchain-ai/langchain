* Default to asynchronous methods (ainvoke, abatch, astream) where applicable.
* Prioritize LangChain Expression Language (LCEL) for creating chains. Use the pipe operator (|) to connect components.
* Generate modular imports from specific packages like langchain_core, langchain_community, and langchain_openai. Do not import from the top-level langchain package.
* Format all generated Python code to be compliant with ruff rules.
* All generated Python code must include type hints.
* When suggesting package installation commands, use uv pip install as this project uses uv.
* When creating tools for agents, use the @tool decorator from langchain_core.tools. The tool's docstring serves as its functional description for the agent.
* Avoid suggesting deprecated components, such as the legacy LLMChain.
* We use Conventional Commits format for pull request titles:

```txt
# Enforced Commit Message Format (Conventional Commits 1.0.0):
#   <type>[optional scope]: <description>
#   [optional body]
#   [optional footer(s)]
#
# Allowed Types:
#   • feat       — a new feature (MINOR bump)
#   • fix        — a bug fix (PATCH bump)
#   • docs       — documentation only changes
#   • style      — formatting, missing semi-colons, etc.; no code change
#   • refactor   — code change that neither fixes a bug nor adds a feature
#   • perf       — code change that improves performance
#   • test       — adding missing tests or correcting existing tests
#   • build      — changes that affect the build system or external dependencies
#   • ci         — continuous integration/configuration changes
#   • chore      — other changes that don't modify src or test files
#   • revert     — reverts a previous commit
#   • release    — prepare a new release
#
# Allowed Scopes (optional):
#   core, cli, langchain, standard-tests, docs, anthropic, chroma, deepseek,
#   exa, fireworks, groq, huggingface, mistralai, nomic, ollama, openai,
#   perplexity, prompty, qdrant, xai
#
# Rules & Tips for New Committers:
#   1. Subject (type) must start with a lowercase letter and, if possible, be
#      followed by a scope wrapped in parenthesis `(scope)`
#   2. Breaking changes:
#        – Append "!" after type/scope (e.g., feat!: drop Node 12 support)
#        – Or include a footer "BREAKING CHANGE: <details>"
#   3. Example PR titles:
#        feat(core): add multi‐tenant support
#        fix(cli): resolve flag parsing error
#        docs: update API usage examples
#        docs(openai): update API usage examples
```
