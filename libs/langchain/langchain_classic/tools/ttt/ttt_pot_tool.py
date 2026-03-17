"""TTT Proof-of-Time tools for LangChain agents.

Provides cryptographic temporal attestation for agent transactions.
Proves ordering without trusting a single time source.

Uses 4 independent HTTPS time sources (NIST, Apple, Google, Cloudflare)
with a 3-layer integrity pipeline for Byzantine resistance.

IETF Draft: draft-helmprotocol-tttps-00
npm: openttt@0.2.0 | GitHub: Helm-Protocol/OpenTTT
"""

import os
from typing import Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import httpx


class TTTPoTInput(BaseModel):
    tx_hash: str = Field(description="Transaction hash to anchor")
    chain_id: int = Field(default=84532, description="Chain ID (84532=Base Sepolia)")


class TTTPoTTool(BaseTool):
    """Generate Proof-of-Time attestations for transactions.

    Use this tool when an agent executes a blockchain transaction
    and needs cryptographic proof of when it occurred.
    """

    name: str = "ttt_pot_generate"
    description: str = (
        "Generate a Proof-of-Time (PoT) attestation before a transaction "
        "hits the chain. Returns a signed timestamp anchor that proves "
        "ordering. Use pot_verify after tx confirms."
    )
    args_schema: type[BaseModel] = TTTPoTInput

    mcp_url: str = Field(
        default_factory=lambda: os.environ.get("TTT_MCP_URL", "http://localhost:3000")
    )

    def _run(self, tx_hash: str, chain_id: int = 84532) -> dict:
        with httpx.Client() as client:
            resp = client.post(
                f"{self.mcp_url}/pot_generate",
                json={"txHash": tx_hash, "chainId": chain_id}
            )
            resp.raise_for_status()
            return resp.json()

    async def _arun(self, tx_hash: str, chain_id: int = 84532) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.mcp_url}/pot_generate",
                json={"txHash": tx_hash, "chainId": chain_id}
            )
            resp.raise_for_status()
            return resp.json()


class TTTPoTVerifyTool(BaseTool):
    """Verify a Proof-of-Time attestation after transaction confirms."""

    name: str = "ttt_pot_verify"
    description: str = (
        "Verify a PoT attestation after transaction lands on-chain. "
        "Returns ordering proof. Use this to detect frontrunning."
    )

    mcp_url: str = Field(
        default_factory=lambda: os.environ.get("TTT_MCP_URL", "http://localhost:3000")
    )

    def _run(self, pot_hash: str) -> dict:
        with httpx.Client() as client:
            resp = client.post(
                f"{self.mcp_url}/pot_verify",
                json={"potHash": pot_hash}
            )
            resp.raise_for_status()
            return resp.json()

    async def _arun(self, pot_hash: str) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.mcp_url}/pot_verify",
                json={"potHash": pot_hash}
            )
            resp.raise_for_status()
            return resp.json()
