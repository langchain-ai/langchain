---
type: "Reference"
title: "CI Workflows (Deprecated)"
description: "Deprecated page: Actual CI workflows documentation has been moved to /openwiki/operations/ci-workflows.md."
tags: [redirect, deprecated, ci, workflows]
sources:
  - id: openwiki-source-c92cc62695c6def991956428
    resource: repo://.github/workflows/_release.yml
  - id: openwiki-source-6e3a52c89729b5704dbd7eec
    resource: repo://.github/workflows/check_diffs.yml
  - id: openwiki-source-9069a5dd5fbb579fbd5470ce
    resource: repo://.github/workflows/integration_tests.yml
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-26T08:33:32.568Z
---

## Deprecated

This page has been deprecated. Please refer to the updated documentation:

**[CI Workflows Documentation](/openwiki/operations/ci-workflows.md)**

The LangChain CI/CD infrastructure documentation has been consolidated into the operations section. See the updated page for comprehensive information about PR CI, scheduled integration tests, release workflows, and change detection systems.

### What Was Here

This page previously documented LangChain's CI/CD infrastructure, which centers on three interconnected GitHub Actions workflows:

1. **PR CI (`check_diffs.yml`)** — Runs on every PR/push to detect changed files and dynamically select affected test suites with intelligent change detection and matrix testing across Python versions and Pydantic variants.

2. **Scheduled Integration Tests (`integration_tests.yml`)** — Daily remote API testing with live credentials to catch breaking changes in external dependencies.

3. **Release Workflow (`_release.yml`)** — Publishes to PyPI with strict release gates that validate packages before publishing.

All details about these workflows, including change detection mechanisms, test matrices, configuration, and operational behavior, are now documented in the operations section.
