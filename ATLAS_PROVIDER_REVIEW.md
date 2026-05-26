# Atlas Cloud Provider Review

## Summary

This change adds a minimal `langchain-atlas` partner package to the LangChain monorepo and wires it into the existing chat model initialization flow.

The implementation follows the same OpenAI-compatible pattern used by existing providers and keeps the scope intentionally small:

- Added a standalone partner package at `libs/partners/atlas`
- Implemented `ChatAtlas` on top of `BaseChatOpenAI`
- Registered the provider in `langchain` and `langchain_v1` chat model factories
- Added serialization allowlist entries for `ChatAtlas`
- Updated repository metadata and workflow configuration so the new package is recognized by CI/release tooling
- Added Atlas Cloud README content and logo references

## Provider Behavior

`ChatAtlas` reads:

- `ATLAS_API_KEY`
- `ATLAS_API_BASE`

Default API base:

```text
https://api.atlascloud.ai/v1
```

Default verified model used in examples and integration tests:

```text
deepseek-ai/DeepSeek-V3-0324
```

## Files Added

- `libs/partners/atlas/langchain_atlas/__init__.py`
- `libs/partners/atlas/langchain_atlas/chat_models.py`
- `libs/partners/atlas/README.md`
- `libs/partners/atlas/.env.example`
- `libs/partners/atlas/assets/atlas-cloud-logo.png`
- `libs/partners/atlas/tests/unit_tests/test_chat_models.py`
- `libs/partners/atlas/tests/unit_tests/test_secrets.py`
- `libs/partners/atlas/tests/unit_tests/__snapshots__/test_chat_models.ambr`
- `libs/partners/atlas/tests/integration_tests/test_chat_models.py`

## Files Updated

- `README.md`
- `libs/README.md`
- `libs/core/langchain_core/load/load.py`
- `libs/core/langchain_core/load/mapping.py`
- `libs/langchain/langchain_classic/chat_models/base.py`
- `libs/langchain/pyproject.toml`
- `libs/langchain/uv.lock`
- `libs/langchain_v1/langchain/chat_models/base.py`
- `libs/langchain_v1/pyproject.toml`
- `libs/langchain_v1/uv.lock`
- `.github/scripts/pr-labeler-config.json`
- `.github/workflows/pr_lint.yml`
- `.github/workflows/_release.yml`
- `.github/workflows/integration_tests.yml`
- `.github/workflows/auto-label-by-package.yml`
- `.github/dependabot.yml`
- `.github/ISSUE_TEMPLATE/bug-report.yml`
- `.github/ISSUE_TEMPLATE/feature-request.yml`
- `.github/ISSUE_TEMPLATE/privileged.yml`
- `.github/ISSUE_TEMPLATE/task.yml`

## Validation

Completed locally:

- Atlas unit tests
- Atlas secret masking test
- Atlas real API invoke test
- Atlas real API streaming test
- Targeted `langchain` chat model regression tests
- Targeted `langchain_v1` chat model regression tests

## Notes

- The local API key was stored in the repository root `.env` for testing and is intentionally not part of the committed changes.
- The integration test file is intentionally minimal and only validates the working Atlas chat path and streaming path.
- No temporary ad hoc test script is included in the commit.
