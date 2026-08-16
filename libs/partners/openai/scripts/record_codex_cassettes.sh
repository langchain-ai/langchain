#!/usr/bin/env bash
# Record VCR cassettes for `ChatOpenAICodex` integration tests against a
# real ChatGPT OAuth subscription account, then verify no OAuth secrets
# survived into the on-disk cassettes.
#
# Usage:
#   scripts/record_codex_cassettes.sh                                 # all integration_tests
#   scripts/record_codex_cassettes.sh tests/integration_tests/chat_models/test_codex.py
#   scripts/record_codex_cassettes.sh tests/integration_tests/chat_models/test_codex.py::test_invoke
#
# Env:
#   CHATGPT_AUTH_FILE   Override the token store path. Defaults to
#                       `$HOME/.langchain/chatgpt-auth.json`.
#   PYTEST_EXTRA        Extra args forwarded to pytest (e.g. `-x -k codex`).
#
# Exits non-zero if the token preflight fails, if pytest fails, or if the
# post-recording leak scan finds OAuth secrets in any cassette.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASSETTE_DIR="${PKG_DIR}/tests/cassettes"
TOKEN_FILE="${CHATGPT_AUTH_FILE:-${HOME}/.langchain/chatgpt-auth.json}"

if [[ ! -f "${TOKEN_FILE}" ]]; then
    echo "error: ChatGPT OAuth token store not found at ${TOKEN_FILE}." >&2
    echo "       Run \`python -c 'from langchain_openai.chatgpt_oauth import login_chatgpt; login_chatgpt()'\` first." >&2
    exit 2
fi

cd "${PKG_DIR}"

# Reserve all tempfiles upfront and register a single cumulative cleanup
# trap so a failure between stages can't leak files (notably the leak
# report, which would contain the very tokens we just scrubbed). Mode 600
# on the leak report seals it against other users sharing the host while
# the script runs.
PRE_SNAPSHOT="$(mktemp)"
POST_SNAPSHOT="$(mktemp)"
LEAK_REPORT="$(mktemp)"
chmod 600 "${LEAK_REPORT}" "${PRE_SNAPSHOT}" "${POST_SNAPSHOT}"
trap 'rm -f "${PRE_SNAPSHOT}" "${POST_SNAPSHOT}" "${LEAK_REPORT}"' EXIT

# Portable cassette snapshot: GNU `find -printf` is unavailable on BSD/macOS
# and BSD `stat -f` is unavailable on Linux. A short python one-liner
# avoids the platform split and surfaces real errors instead of swallowing
# them behind `|| true`.
snapshot_cassettes() {
    local dest="$1"
    if [[ ! -d "${CASSETTE_DIR}" ]]; then
        : > "${dest}"
        return 0
    fi
    python - "${CASSETTE_DIR}" >"${dest}" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
entries = []
for path in root.rglob("*.yaml.gz"):
    try:
        mtime = path.stat().st_mtime
    except OSError as exc:
        print(f"warning: failed to stat {path}: {exc}", file=sys.stderr)
        continue
    entries.append(f"{path} {mtime}")
entries.sort()
sys.stdout.write("\n".join(entries))
if entries:
    sys.stdout.write("\n")
PY
}

# Preflight: force a refresh now so VCR doesn't record an `auth.openai.com`
# token roundtrip mid-test. A stale or revoked refresh token surfaces here
# rather than after a long test run.
echo "==> Refreshing ChatGPT OAuth token (${TOKEN_FILE})"
CHATGPT_AUTH_FILE="${TOKEN_FILE}" uv run --group test python - <<'PY'
import os
import sys
from pathlib import Path

from langchain_openai.chatgpt_oauth import (
    ChatGPTOAuthRefreshError,
    FileChatGPTOAuthTokenProvider,
)

path = Path(os.environ["CHATGPT_AUTH_FILE"])
provider = FileChatGPTOAuthTokenProvider(path=path)
try:
    token = provider.get_token()
except ChatGPTOAuthRefreshError as exc:
    print(f"refresh failed: {exc}", file=sys.stderr)
    sys.exit(3)
print(f"ok — token valid until {token.expires_at.isoformat()}")
PY

snapshot_cassettes "${PRE_SNAPSHOT}"

# Default target: full integration suite. Override with positional args.
if [[ "$#" -eq 0 ]]; then
    set -- tests/integration_tests/
fi

# `ChatOpenAICodex` constructs an `api_key` callable from its token provider,
# but `ChatOpenAI`'s pydantic init still requires *some* non-empty value.
# The placeholder is intentionally < 20 chars after `sk-` so the leak-scan
# pattern below (which targets real API keys at >= 20 chars) doesn't
# false-positive on it.
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-codex-placeholder}"

echo "==> Recording cassettes for: $*"
# `--record-mode=once` writes any missing cassette and replays existing ones.
# `-m vcr` limits the run to VCR-marked tests so unrelated live tests don't
# fire unexpectedly. `PYTEST_EXTRA` is word-split via the shell (the
# `disable=SC2086` is intentional) so embedded spaces split into separate
# args — pass complex flags (e.g., `-k "name with spaces"`) as positional
# args to the script itself instead.
# shellcheck disable=SC2086
uv run --group test --group test_integration pytest \
    --record-mode=once \
    -m vcr \
    -v --tb=short \
    ${PYTEST_EXTRA:-} \
    "$@"

# Leak scan: zgrep the post-state cassettes for any pattern that would
# indicate an OAuth secret slipped past the conftest scrubbers. All
# patterns are passed to a single zgrep invocation per file so each
# cassette is decompressed exactly once. Patterns are ERE (zgrep -E):
# brace quantifiers use `{N,}` without backslashes.
echo "==> Scanning cassettes for OAuth secret leaks"
LEAK_ZGREP_ARGS=(
    -e 'Bearer ey'                                        # bearer token in a captured header value
    -e 'eyJ[A-Za-z0-9_-]{20,}\.'                          # JWT-shaped payload (access/id/refresh tokens)
    -e '"refresh_token"[[:space:]]*:[[:space:]]*"[^*"]'
    -e '"access_token"[[:space:]]*:[[:space:]]*"[^*"]'
    -e '"id_token"[[:space:]]*:[[:space:]]*"[^*"]'
    -e 'refresh_token=[^&*]'                              # urlencoded refresh-grant body
    -e 'sk-[A-Za-z0-9]{20,}'                              # leaked API key (>= 20 chars after sk-)
    -e 'chatgpt_account_id'                               # account-id JWT claim from an id_token payload
)

leak_found=0
if [[ -d "${CASSETTE_DIR}" ]]; then
    while IFS= read -r -d '' cassette; do
        if zgrep -aHE "${LEAK_ZGREP_ARGS[@]}" "${cassette}" >> "${LEAK_REPORT}" 2>/dev/null; then
            leak_found=1
        fi
    done < <(find "${CASSETTE_DIR}" -type f -name '*.yaml.gz' -print0)
fi

if [[ "${leak_found}" -ne 0 ]]; then
    echo "error: OAuth secret leak detected in cassettes:" >&2
    cat "${LEAK_REPORT}" >&2
    echo >&2
    echo "Do NOT commit these cassettes. Re-run after extending the" >&2
    echo "scrubber in tests/conftest.py to cover the leaking field." >&2
    exit 4
fi

snapshot_cassettes "${POST_SNAPSHOT}"

# Summarize what changed so the user knows which cassettes to inspect.
# Capture diff output to a variable so we can branch on `diff`'s exit
# code rather than on a piped one (`set -o pipefail` would otherwise
# flip the sense of the test).
echo "==> Cassette changes:"
if diff_out=$(diff -u "${PRE_SNAPSHOT}" "${POST_SNAPSHOT}"); then
    echo "(no changes detected)"
else
    # Strip the unified-diff `---`/`+++` headers; keep the `+`/`-` body lines.
    printf '%s\n' "${diff_out}" | grep -E '^[+-][^+-]' || true
fi

echo
echo "Recording complete. Inspect the diff before committing:"
echo "    git -C ${PKG_DIR} status tests/cassettes/"
echo "    git -C ${PKG_DIR} diff --stat tests/cassettes/"
