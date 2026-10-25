---
type: Redirect
title: Integration Tests (Deprecated Reference)
description: This page has been moved. For the authoritative documentation on integration testing, VCR cassettes, and standard test suites, see the Integration Testing documentation.
tags: [integration-tests, redirect, deprecated]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
sources:
  - id: openwiki-source-9069a5dd5fbb579fbd5470ce
    resource: repo://.github/workflows/integration_tests.yml
  - id: openwiki-source-711811fb8df6b32329091d23
    resource: repo://libs/partners/openai/tests/integration_tests/chat_models/test_base_standard.py
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

## Deprecated Page

This page has been superseded. For the authoritative documentation on integration testing, VCR cassettes, test markers, and standard test suite inheritance, please see:

**→ [Integration Testing: Standard Test Suites & Providers](/openwiki/integration-tests.md)**

The primary page contains comprehensive coverage of:

- **Test Markers** – `@pytest.mark.vcr` for cassette-backed tests, `@pytest.mark.scheduled` for live API testing
- **VCR Cassettes** – Recording, playback, cassette location, naming conventions, and scrubbing sensitive data
- **Cassette Security** – Header filtering, body scrubbing, and credential protection
- **Standard Test Suite Inheritance** – `ChatModelIntegrationTests`, customization points, and provider-specific overrides
- **CI/CD Integration** – Compilation checks, VCR playback in PR checks, and scheduled daily workflows
- **Environment Setup** – `.env` configuration, API key management, and local testing
- **Cassette Management** – Updating cassettes, rotating credentials, and version upgrades
- **Best Practices** – Test isolation, mocking, determinism, and debugging
- **Troubleshooting** – Common issues and solutions for cassette playback and recording
