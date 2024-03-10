Thank you for contributing to LangChain!

- [ ] **PR title**: "package: description"
  - Where "package" is whichever of langchain, community, core, experimental, etc. is being modified. Use "docs: ..." for purely docs changes, "templates: ..." for template changes, "infra: ..." for CI changes.
  - Example: "community: add foobar LLM"


- [ ] **PR message**: ***Delete this entire checklist*** and replace with
    - **Description:** a description of the change
    - **Issue:** the issue # it fixes, if applicable
    - **Dependencies:** any dependencies required for this change
    - **Twitter handle:** if your PR gets announced, and you'd like a mention, we'll gladly shout you out!


- [ ] **Add tests and docs**: If you're adding a new integration, please include
  1. a test for the integration, preferably unit tests that do not rely on network access,
  2. an example notebook showing its use. It lives in `docs/docs/integrations` directory.


- [ ] **Lint and test**: Run `make format`, `make lint` and `make test` from the root of the package(s) you've modified. See contribution guidelines for more: https://python.langchain.com/docs/contributing/

Additional guidelines:
- Make sure optional dependencies are imported within a function.
- Please do not add dependencies to pyproject.toml files (even optional ones) unless they are required for unit tests.
- Most PRs should not touch more than one package.
- Changes should be backwards compatible.
- If you are adding something to community, do not re-import it in langchain.

If no one reviews your PR within a few days, please @-mention one of baskaryan, efriis, eyurtsev, hwchase17.
