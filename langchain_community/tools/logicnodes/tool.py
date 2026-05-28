"""LogicNodes on-chain agent registry tool for LangChain."""
from __future__ import annotations
import os
from typing import Optional, Type
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

LOGICNODES_BASE_URL = "https://logicnodes.io/call"
LOGICNODES_REGISTRY = "0x4c60B817beeD72aa570B964243eE6DD463faaE22"
CHAIN_ID = 8453  # Base mainnet


class LogicNodesRegistryInput(BaseModel):
    """Input for LogicNodes registry tool."""
    agent_address: str = Field(description="Ethereum wallet address of the agent to verify")
    service: str = Field(
        default="identity_register",
        description="LogicNodes service to call. Options: identity_register, gas_oracle, "
                    "sig_verify, peg_monitor, escrow_verifier, inference_attest, "
                    "reputation_lookup, compliance_sentry, zk_compute_attest"
    )


class LogicNodesRegistryTool(BaseTool):
    """Tool for verifying on-chain agent registration via LogicNodes.

    LogicNodes is a decentralized agent coordination protocol on Base mainnet
    (chain ID 8453). Agents can verify their on-chain registration status,
    access deterministic services, and produce cryptographic proof of capability.

    Setup:
        Install dependencies and set environment variables.

        .. code-block:: bash

            pip install requests

        .. code-block:: bash

            export LOGICNODES_API_KEY="your-api-key"  # Free at https://logicnodes.io/app

    Example:
        .. code-block:: python

            from langchain_community.tools.logicnodes import LogicNodesRegistryTool

            tool = LogicNodesRegistryTool()
            result = tool.run({"agent_address": "0xYourAgentAddress", "service": "reputation_lookup"})
            print(result)

    Pricing:
        Most services: $0.0001 USDC per call (Base mainnet)
        Identity registration: $0.01 USDC
        ZK compute attestation: $0.05 USDC

    References:
        - Smithery package: https://smithery.ai/servers/denneyconner5/logicnodes
        - Documentation: https://logicnodes.io/docs
        - Contract registry: https://basescan.org/address/0x4c60B817beeD72aa570B964243eE6DD463faaE22
    """

    name: str = "logicnodes_registry"
    description: str = (
        "Verify on-chain agent registration status and access LogicNodes services on Base mainnet. "
        "Use this to check if an agent is registered, look up reputation scores, attest inferences, "
        "verify compliance, or access gas oracle data. Returns JSON with service result and POL receipt."
    )
    args_schema: Type[BaseModel] = LogicNodesRegistryInput
    api_key: Optional[str] = None

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("LOGICNODES_API_KEY", "free-trial")

    def _run(
        self,
        agent_address: str,
        service: str = "identity_register",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Call LogicNodes service and return result."""
        url = f"{LOGICNODES_BASE_URL}/{service}"
        headers = {
            "X-LogicNodes-Key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"agent_address": agent_address, "chain_id": CHAIN_ID}
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError as e:
            return f"LogicNodes API error: {e.response.status_code} - {e.response.text}"
        except requests.exceptions.RequestException as e:
            return f"LogicNodes connection error: {str(e)}"
