"""X-Trust human presence attestation callback for LangChain."""
from __future__ import annotations
import base64, hashlib, hmac, json, time
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


def _b64url_to_bytes(s: str) -> bytes:
    s = s.replace("-", "+").replace("_", "/")
    pad = 4 - len(s) % 4
    if pad != 4:
        s += "=" * pad
    return base64.b64decode(s)


def verify_x_trust(token: str, secret: str) -> Optional[Dict[str, Any]]:
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != "v1":
        return None
    _, payload_b64, sig_b64 = parts
    try:
        expected = hmac.new(
            secret.encode(), payload_b64.encode(), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(_b64url_to_bytes(sig_b64), expected):
            return None
        payload = json.loads(_b64url_to_bytes(payload_b64))
        now = int(time.time())
        if not isinstance(payload.get("score"), (int, float)):
            return None
        if not (0 <= payload["score"] <= 1):
            return None
        if now > payload.get("exp", 0) or now - payload.get("iat", 0) > 120:
            return None
        return payload
    except Exception:
        return None


class XTrustCallbackHandler(BaseCallbackHandler):
    """Annotates LangChain runs with X-Trust human presence score.

    Implements the AIR doctrine: Annotate, never block.

    Example:
        .. code-block:: python

            from langchain_community.callbacks import XTrustCallbackHandler
            from langchain_openai import ChatOpenAI

            handler = XTrustCallbackHandler(
                secret=os.environ["HTL_SECRET"],
                x_trust_token=request.headers.get("x-trust", ""),
            )
            llm = ChatOpenAI(callbacks=[handler])
            response = llm.invoke("Hello!")
            print(handler.trust_score)  # e.g. 0.88
            print(handler.trusted)      # True
    """

    def __init__(
        self,
        secret: str,
        x_trust_token: str = "",
        min_score: float = 0.0,
    ) -> None:
        self.secret = secret
        self.x_trust_token = x_trust_token
        self.min_score = min_score
        self.trust_score: float = 0.0
        self.trusted: bool = False
        self.annotated: bool = True

        payload = verify_x_trust(x_trust_token, secret) if x_trust_token else None
        if payload:
            self.trust_score = payload["score"]
            self.trusted = self.trust_score >= min_score

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        if isinstance(kwargs.get("metadata"), dict):
            kwargs["metadata"]["x_trust_score"] = self.trust_score
            kwargs["metadata"]["x_trust_annotated"] = self.annotated

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass
