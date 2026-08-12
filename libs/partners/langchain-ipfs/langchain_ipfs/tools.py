"""Tool for pinning files to IPFS via the ipfs-pay-to-pin service."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field, model_validator


class IPFSPinToolInput(BaseTool):  # type: ignore[override]
    """Input schema for IPFSPinTool."""

    content: str = Field(
        description=(
            "The text content to pin to IPFS. The content will be converted "
            "to a file and uploaded to the IPFS network via the ipfs-pay-to-pin service."
        ),
    )
    pin_name: str | None = Field(
        None,
        description=(
            "Optional human-readable name for the pin. Defaults to 'ipfs-pin-{timestamp}'."
        ),
    )
    expiration_days: int | None = Field(
        None,
        description=(
            "Retention period in days (1-365). Defaults to 30. "
            "The pin will be automatically removed after this period."
        ),
    )


class IPFSPinTool(BaseTool):  # type: ignore[override]
    """Tool for pinning content to IPFS via the ipfs-pay-to-pin service.

    Setup:
        Install `langchain-ipfs` and `ipfs-pay-to-pin-client`:

        .. code-block:: bash

            pip install langchain-ipfs ipfs-pay-to-pin-client

        Set your API key in the environment:

        .. code-block:: bash

            export IPFS_PIN_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_ipfs import IPFSPinTool

            tool = IPFSPinTool()

    Invocation with args:
        .. code-block:: python

            tool.invoke({"content": "Hello, IPFS!", "pin_name": "my-first-pin"})

    Example:
        .. code-block:: python

            from langchain_ipfs import IPFSPinTool

            tool = IPFSPinTool(
                chain="base",  # Base L2 (gasless, default)
                max_price_usdc=0.01  # Max $0.01 per pin
            )
            result = tool.invoke({
                "content": "Important content to pin",
                "pin_name": "my-document",
                "expiration_days": 90
            })

    Note:
        This tool requires the ipfs-pay-to-pin-client package. Install it with:

        .. code-block:: bash

            pip install ipfs-pay-to-pin-client

    .. versionadded:: 0.1.0
    """

    name: str = "ipfs_pin_tool"
    description: str = (
        "A tool that pins content to IPFS via the ipfs-pay-to-pin service. "
        "Uploads text content to the InterPlanetary File System (IPFS) and returns "
        "a Content Identifier (CID) for permanent retrieval. Supports configurable "
        "pin names and retention periods (1-365 days). Uses x402 micropayments for "
        "decentralized pinning across multiple blockchains (Base L2, Solana, "
        "Algorand, Ethereum L1). To use this tool, provide 'content' with the text to pin."
    )
    args_schema: type[BaseTool] = IPFSPinToolInput
    api_key: str | None = None
    base_url: str | None = None
    chain: str | None = None
    max_price_usdc: float | None = None
    client: Any = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Initialize the client from config or environment."""
        return values  # Client init happens in _run

    def _run(
        self,
        content: str,
        pin_name: str | None = None,
        expiration_days: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Pin content to IPFS.

        Args:
            content: The text content to pin.
            pin_name: Optional name for the pin.
            expiration_days: Retention period (1-365 days).
            run_manager: Unused, for BaseTool compatibility.

        Returns:
            A string containing the result with CID and metadata, or an error message.
        """
        try:
            from ipfs_pay_to_pin_client import PinClient, PinRequest, PinResponse

            # Initialize client if not already done
            if self.client is None:
                client_kwargs: dict[str, Any] = {}
                if self.api_key:
                    client_kwargs["api_key"] = self.api_key
                elif pin_name is not None:  # Placeholder to avoid unused var warning
                    pass
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                if self.chain:
                    client_kwargs["chain"] = self.chain
                if self.max_price_usdc is not None:
                    client_kwargs["max_price_usdc"] = self.max_price_usdc

                self.client = PinClient(**client_kwargs)

            response: PinResponse = self.client.pin(
                PinRequest(
                    content=content,
                    pin_name=pin_name,
                    expiration_days=expiration_days,
                )
            )

            result_lines = [
                "Successfully pinned to IPFS!",
                f"CID: {response.cid}",
                f"Pin name: {response.pin_name or 'N/A'}",
            ]

            if response.size_bytes:
                result_lines.append(f"Size: {response.size_bytes} bytes")
            if response.pin_id:
                result_lines.append(f"Pin ID: {response.pin_id}")
            if response.status:
                result_lines.append(f"Status: {response.status}")

            return "\n".join(result_lines)

        except ImportError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: Failed to pin content to IPFS. {str(e)}"
