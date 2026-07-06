import inspect
import json
import logging
import re
from typing import Any

import pytest
from langchain_tests.conftest import CustomPersister, CustomSerializer, base_vcr_config
from vcr import VCR  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_EXTRA_HEADERS = [
    ("openai-organization", "PLACEHOLDER"),
    ("user-agent", "PLACEHOLDER"),
    ("x-openai-client-user-agent", "PLACEHOLDER"),
    # ChatGPT OAuth subscription auth: the catch-all redactor below already
    # wipes every header, but list these explicitly so anyone reading the
    # config knows they are covered.
    ("chatgpt-account-id", "PLACEHOLDER"),
    ("cookie", "PLACEHOLDER"),
    ("set-cookie", "PLACEHOLDER"),
]

# OAuth secret-bearing fields. Redacted in request and response bodies as
# defense-in-depth against an OAuth token-endpoint roundtrip getting captured
# mid-cassette.
_OAUTH_SECRET_FIELDS = frozenset(
    {
        "access_token",
        "refresh_token",
        "id_token",
        "code",
        "code_verifier",
        "device_code",
        "client_secret",
    }
)
_JWT_PATTERN = re.compile(rb"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")

# Binary content magic bytes. If any of these prefix the body we skip the
# scrub stack — JWTs (and every other OAuth secret we redact) are ASCII,
# so a confirmed-binary body can't carry one. Matters for performance:
# audio/PDF/image cassette responses can be many hundreds of KB and the
# UTF-8 + regex passes are O(N) on every record/replay.
_BINARY_MAGIC_PREFIXES: tuple[bytes, ...] = (
    b"\x89PNG",  # PNG
    b"\xff\xd8\xff",  # JPEG
    b"GIF8",  # GIF
    b"RIFF",  # WAV / WEBP container
    b"OggS",  # Ogg
    b"ID3",  # MP3 with ID3 tag
    b"\xff\xfb",  # MP3 without tag
    b"%PDF",  # PDF
    b"PK\x03\x04",  # ZIP / DOCX / etc.
)


def _scrub_form_body(body: bytes) -> bytes:
    """Redact OAuth secret fields in a urlencoded form body."""
    text = body.decode("utf-8", errors="replace")
    parts = text.split("&")
    redacted: list[str] = []
    for part in parts:
        key, sep, _ = part.partition("=")
        if sep and key in _OAUTH_SECRET_FIELDS:
            redacted.append(f"{key}=**REDACTED**")
        else:
            redacted.append(part)
    return "&".join(redacted).encode("utf-8")


def _walk_and_redact(node: Any) -> bool:
    """Redact OAuth secret fields anywhere in a parsed JSON tree in place.

    Returns `True` if any field was redacted. Walks dicts and lists
    recursively so nested OAuth payloads (e.g., `{"data": {"refresh_token":
    "..."}}` or `[{"access_token": "..."}]`) are scrubbed too.
    """
    redacted = False
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _OAUTH_SECRET_FIELDS and isinstance(value, (str, int, float)):
                node[key] = "**REDACTED**"
                redacted = True
            elif isinstance(value, (dict, list)):
                redacted = _walk_and_redact(value) or redacted
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                redacted = _walk_and_redact(item) or redacted
    return redacted


def _scrub_json_body(body: bytes) -> tuple[bytes, bool]:
    """Redact OAuth secrets in a JSON body recursively.

    Returns `(scrubbed_bytes, was_json)`. `was_json` is `True` whenever the
    body parsed successfully, so callers can skip the form-body fallback
    (form-decoding a JSON body would split on `&` in string values and
    mangle them). The returned bytes match the input byte-for-byte when no
    field was redacted, so unrelated cassette diffs don't churn from JSON
    whitespace differences.
    """
    try:
        payload = json.loads(body)
    except (ValueError, json.JSONDecodeError):
        return body, False
    if not isinstance(payload, (dict, list)):
        return body, True
    if not _walk_and_redact(payload):
        return body, True
    return json.dumps(payload).encode("utf-8"), True


def _scrub_oauth_secrets(body: bytes | str | None) -> bytes | str | None:
    """Best-effort scrubber for OAuth secrets in request/response bodies.

    Text bodies (JSON or urlencoded form) get a full structured scrub.
    Bodies that begin with a known binary magic prefix (PNG, JPEG, PDF,
    audio, etc.) pass through untouched — JWTs and the rest of
    `_OAUTH_SECRET_FIELDS` are ASCII, so a confirmed-binary payload can't
    carry one. Skipping these saves O(N) UTF-8 + regex work on cassette
    responses that can run into the hundreds of KB.

    Partially-malformed UTF-8 (a corrupted text response) still gets the
    JWT regex pass via `errors="replace"`, so a token-endpoint reply with
    a single bad byte can't slip a JWT past us.
    """
    if not body:
        return body
    if isinstance(body, bytes):
        if body.startswith(_BINARY_MAGIC_PREFIXES):
            return body
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            # Not text and no known binary prefix — could be a partially-
            # malformed text response. Run the JWT regex over a lossy decode
            # so a token-endpoint reply with a bad byte can't sneak through.
            # The lossy decode is *only* used for matching — the original
            # bytes are returned if no JWT was found.
            fallback_text = body.decode("utf-8", errors="replace").encode("utf-8")
            scrubbed_fallback = _JWT_PATTERN.sub(b"**REDACTED-JWT**", fallback_text)
            if scrubbed_fallback == fallback_text:
                return body
            logger.warning(
                "Scrubbed a JWT-shaped token from a non-UTF-8 response body "
                "(%d bytes). Originating endpoint likely returned text "
                "with a bad byte; recording was preserved via lossy decode.",
                len(body),
            )
            return scrubbed_fallback
    else:
        text = body
    payload_bytes = text.encode("utf-8")
    scrubbed, was_json = _scrub_json_body(payload_bytes)
    if not was_json:
        scrubbed = _scrub_form_body(payload_bytes)
    # Final pass: blanket-redact any JWT-shaped string that survived (e.g.
    # JWT embedded in a free-form error message).
    scrubbed = _JWT_PATTERN.sub(b"**REDACTED-JWT**", scrubbed)
    return scrubbed.decode("utf-8") if isinstance(body, str) else scrubbed


def remove_request_headers(request: Any) -> Any:
    """Remove sensitive headers and OAuth secrets from the request."""
    for k in request.headers:
        request.headers[k] = "**REDACTED**"
    request.uri = "**REDACTED**"
    request.body = _scrub_oauth_secrets(request.body)
    return request


def remove_response_headers(response: dict) -> dict:
    """Remove sensitive headers and OAuth secrets from the response."""
    for k in response["headers"]:
        response["headers"][k] = "**REDACTED**"
    # Pinning vcrpy's internal `body["string"]` shape: if vcrpy ever
    # switches to a different key the scrub silently no-ops, so the
    # script's post-recording leak scan is the load-bearing backstop.
    body = response.get("body")
    if isinstance(body, dict):
        body_value = body.get("string")
        if body_value is not None:
            body["string"] = _scrub_oauth_secrets(body_value)
    return response


@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Extend the default configuration coming from langchain_tests."""
    config = base_vcr_config()
    config["match_on"] = [
        m if m != "body" else "json_body"
        for m in config.get("match_on", [])
        if m != "uri"
    ]
    config.setdefault("filter_headers", []).extend(_EXTRA_HEADERS)
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"
    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")
    return config


def _normalize_tool_descriptions(body: Any) -> Any:
    """Strip common leading whitespace from `tools[*].description` strings.

    Python 3.13+ runs `inspect.cleandoc` on docstrings at compile time, but
    cassettes recorded on Python 3.12 (or earlier) preserve the raw indented
    form. Normalize the `description` field — populated from a tool's
    docstring by `@tool` — so cassettes recorded on any Python version match
    requests issued by any other version. Scoped to `tools[*].description`
    to avoid mutating user-visible message content.
    """
    if not isinstance(body, dict):
        return body
    tools = body.get("tools")
    if isinstance(tools, list):
        for entry in tools:
            if isinstance(entry, dict):
                description = entry.get("description")
                if isinstance(description, str):
                    entry["description"] = inspect.cleandoc(description)
    return body


def _json_body_matcher(r1: Any, r2: Any) -> None:
    """Match request bodies as parsed JSON, ignoring key order."""
    b1 = r1.body or b""
    b2 = r2.body or b""
    if isinstance(b1, bytes):
        b1 = b1.decode("utf-8")
    if isinstance(b2, bytes):
        b2 = b2.decode("utf-8")
    try:
        j1 = _normalize_tool_descriptions(json.loads(b1))
        j2 = _normalize_tool_descriptions(json.loads(b2))
    except (json.JSONDecodeError, ValueError):
        assert b1 == b2, f"body mismatch (non-JSON):\n{b1}\n!=\n{b2}"
        return
    assert j1 == j2, f"body mismatch:\n{j1}\n!=\n{j2}"


def pytest_recording_configure(config: dict, vcr: VCR) -> None:
    vcr.register_persister(CustomPersister())
    vcr.register_serializer("yaml.gz", CustomSerializer())
    vcr.register_matcher("json_body", _json_body_matcher)
