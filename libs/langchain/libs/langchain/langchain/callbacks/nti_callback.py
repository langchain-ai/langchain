import json
import uuid
from typing import Any, Dict
from langchain_core.callbacks import BaseCallbackHandler
from ube_foundation import TrustEngine, PqcKeyPair

class NTICallbackHandler(BaseCallbackHandler):
    """
    NTI Callback Handler for LangChain.
    Intercepts tool starts, evaluates zero-trust policy, and blocks unauthorized actions.
    """
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.engine = TrustEngine()
        self.pqc_key = PqcKeyPair.generate()

    def grant_capability(self, capability: str):
        self.engine.grant(self.agent_id, capability)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        tool_input = {"raw_input": input_str}
        
        req = {
            "id": f"req-{uuid.uuid4()}",
            "actor": self.agent_id,
            "capability": tool_name,
            "action": tool_name,
            "input": tool_input,
            "signature": None,
            "pqc_signature": None,
            "public_key": None,
            "pqc_public_key": None,
            "token": None,
            "identity_claim": None
        }
        message = json.dumps(req, sort_keys=True).encode('utf-8')
        req["pqc_signature"] = self.pqc_key.sign(message)
        req["pqc_public_key"] = self.pqc_key.public_key_hex()
        
        decision = json.loads(self.engine.evaluate(json.dumps(req)))
        if decision.get("decision") != "Allow":
            raise PermissionError(f"NTI Security Denied Action: {decision.get('reason')}")
