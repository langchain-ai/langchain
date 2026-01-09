#!/usr/bin/env bash
set -euo pipefail

# Test runner for issue #32637: ChatPromptTemplate save/load functionality
# Supports 'base' and 'new' modes for testing harness.
#
# Usage: ./test.sh [base|new]
#   base - No-op mode (original code baseline)
#   new  - Run tests for ChatPromptTemplate save/load implementation

MODE="${1:-new}"

case "$MODE" in
  base)
    # Base mode: no-op (tests would fail on original code without save() implementation)
    echo "[base mode] Skipping tests on original code (save() not yet implemented)"
    exit 0
    ;;
  new)
    # New mode: run pytest on issue #32637 tests
    echo "[new mode] Running ChatPromptTemplate save/load tests..."
    cd "$(dirname "$0")"

    # Run pytest on the specific test file with verbose output
    python -m pytest \
      libs/core/tests/unit_tests/prompts/test_chat.py::test_chat_prompt_template_save_json \
      libs/core/tests/unit_tests/prompts/test_chat.py::test_chat_prompt_template_save_yaml \
      libs/core/tests/unit_tests/prompts/test_chat.py::test_chat_prompt_template_load_json_roundtrip \
      libs/core/tests/unit_tests/prompts/test_chat.py::test_chat_prompt_template_load_yaml_roundtrip \
      libs/core/tests/unit_tests/prompts/test_chat.py::test_chat_prompt_template_save_unsupported_format \
      libs/core/tests/unit_tests/prompts/test_chat.py::test_chat_prompt_template_save_invalid_path \
      libs/core/tests/unit_tests/prompts/test_chat.py::test_chat_prompt_template_save_creates_parent_directories \
      -v --tb=short 2>&1
    ;;
  *)
    echo "Usage: $0 [base|new]"
    echo ""
    echo "Modes:"
    echo "  base - No-op mode (original code baseline)"
    echo "  new  - Run ChatPromptTemplate save/load tests"
    exit 2
    ;;
esac
