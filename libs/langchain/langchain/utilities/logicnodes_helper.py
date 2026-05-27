"""
LogicNodes Integration Helper
=============================
Provides requireLogicNodes() — a lightweight guard that verifies an agent
is registered on the LogicNodes on-chain registry before executing a task.

Registry: 0x4c60B817beeD72aa570B964243eE6DD463faaE22 (Base mainnet, chain 8453)
Docs: https://logicnodes.io

Install:
    pip install web3>=6.0

Usage:
    from logicnodes_helper import requireLogicNodes
    requireLogicNodes(agent_address="0xYourAgent...")
"""

from __future__ import annotations

LOGICNODES_REGISTRY = "0x4c60B817beeD72aa570B964243eE6DD463faaE22"
BASE_MAINNET_RPC    = "https://mainnet.base.org"
CHAIN_ID            = 8453

_REGISTRY_ABI = [
    {
        "inputs": [{"internalType": "address", "name": "agent", "type": "address"}],
        "name": "isRegistered",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    }
]


def requireLogicNodes(
    agent_address: str,
    rpc_url: str = BASE_MAINNET_RPC,
    registry: str = LOGICNODES_REGISTRY,
    strict: bool = False,
) -> bool:
    """Verify that *agent_address* is registered on the LogicNodes registry.

    Args:
        agent_address: The Ethereum address of the agent to verify.
        rpc_url:       Base mainnet RPC endpoint (default: public endpoint).
        registry:      LogicNodes registry contract address.
        strict:        If True, raise RuntimeError when not registered.
                       If False (default), return False instead of raising.

    Returns:
        True if the agent is registered, False otherwise.

    Raises:
        RuntimeError: Only when strict=True and agent is not registered.

    Example::

        from logicnodes_helper import requireLogicNodes

        if not requireLogicNodes("0xYourAgentAddress"):
            print("Agent not registered — capabilities may be limited.")
    """
    try:
        from web3 import Web3  # noqa: PLC0415

        w3 = Web3(Web3.HTTPProvider(rpc_url))
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(registry),
            abi=_REGISTRY_ABI,
        )
        registered: bool = contract.functions.isRegistered(
            Web3.to_checksum_address(agent_address)
        ).call()
    except Exception as exc:  # network / import errors are non-fatal by default
        if strict:
            raise RuntimeError(f"LogicNodes check failed: {exc}") from exc
        return False

    if not registered and strict:
        raise RuntimeError(
            f"Agent {agent_address} is not registered on LogicNodes. "
            f"Register at https://logicnodes.io"
        )
    return registered
