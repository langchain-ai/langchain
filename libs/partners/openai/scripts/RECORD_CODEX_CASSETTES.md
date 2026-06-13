# Recording `ChatOpenAICodex` VCR cassettes

`ChatOpenAICodex` authenticates with a ChatGPT subscription OAuth bundle, so
its integration tests cannot run in PR CI without a live login. The workflow
below records cassettes once locally with a real token, scrubs OAuth secrets
before they hit disk, and commits the cassettes so CI replays them through
`_test_vcr.yml` (the `vcr-tests` job in `check_diffs.yml`).

## Prerequisites

1. A ChatGPT subscription (Plus / Pro / Team / Enterprise) — required for
   `chatgpt.com/backend-api/codex`.
2. A token bundle on disk. Generate one with:

   ```bash
   uv run --group test python -c \
     "from langchain_openai.chatgpt_oauth import login_chatgpt; login_chatgpt()"
   ```

   The default store is `~/.langchain/chatgpt-auth.json`. It is intentionally
   distinct from `~/.codex/auth.json` so the Codex CLI / VS Code session is
   not invalidated by refresh-token rotation here.
3. An integration test that instantiates `ChatOpenAICodex` and is marked
   `@pytest.mark.vcr`. Tests written against the API-key `ChatOpenAI` will
   not exercise the Codex backend — only Codex-specific tests should be
   passed to the script.

## Record

From `libs/partners/openai/`:

```bash
# Record every VCR-marked integration test (default).
scripts/record_codex_cassettes.sh

# Record one file or one test.
scripts/record_codex_cassettes.sh tests/integration_tests/chat_models/test_codex.py
scripts/record_codex_cassettes.sh \
    tests/integration_tests/chat_models/test_codex.py::test_invoke

# Forward extra pytest args.
PYTEST_EXTRA="-k streaming -x" scripts/record_codex_cassettes.sh
```

The script:

1. Verifies the token store exists.
2. Force-refreshes the access token *outside* pytest so VCR never sees the
   `auth.openai.com/oauth/token` roundtrip. A revoked refresh token surfaces
   here, not after a long recording run.
3. Runs `pytest --record-mode=once -m vcr <target>` so missing cassettes are
   created and existing ones replayed.
4. `zgrep`s every cassette for bearer tokens, JWTs, refresh-grant bodies,
   leaked API keys, and ChatGPT account-id claims. Any match aborts with
   a non-zero exit and a per-file report — do **not** commit those cassettes.
5. Prints a diff of cassette files that were touched.

## What gets scrubbed automatically

`tests/conftest.py` redacts:

- Every request and response header (catches `Authorization: Bearer …`,
  `ChatGPT-Account-Id`, cookies, organization IDs).
- Request URIs (no per-account URL parameters land in cassettes).
- OAuth secret fields in JSON request/response bodies: `access_token`,
  `refresh_token`, `id_token`, `code`, `code_verifier`, `device_code`,
  `client_secret`.
- The same fields in urlencoded form bodies (refresh-grant POSTs).
- Any JWT-shaped string anywhere in a body (`eyJ…`).

The recording script's post-scan exists to catch anything the above misses.

## Override the token store

```bash
CHATGPT_AUTH_FILE=/tmp/codex-test-token.json \
    scripts/record_codex_cassettes.sh
```

Useful when recording against a dedicated test account so refresh rotation
doesn't churn your personal `~/.langchain/chatgpt-auth.json`.

## Commit

```bash
git -C libs/partners/openai status tests/cassettes/
git -C libs/partners/openai diff --stat tests/cassettes/
```

Spot-check at least one new cassette: `gunzip -c tests/cassettes/<name>.yaml.gz | less`.
Verify every `Authorization` header reads `**REDACTED**` and no `eyJ…` strings
remain.

## CI playback

`.github/workflows/check_diffs.yml` routes openai changes through
`_test_vcr.yml`, which runs:

```bash
make test_vcr   # uv run pytest --record-mode=none -m vcr tests/integration_tests/
```

`--record-mode=none` makes pytest fail rather than make a live network call,
so a missing or stale cassette is a hard failure — exactly the signal you
want.
