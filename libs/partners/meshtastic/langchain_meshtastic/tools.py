"""Tool for sending messages via Meshtastic LoRa mesh network devices."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from meshtastic.serial_interface import SerialInterface

__all__ = ["MeshtasticSendInput", "MeshtasticSendTool"]


def _get_meshtastic_serial_interface() -> type[SerialInterface]:
    """Import and return the SerialInterface class from meshtastic.

    Returns:
        The SerialInterface class for creating connections to Meshtastic devices.

    Raises:
        ImportError: If the meshtastic package is not installed.
    """
    try:
        from meshtastic.serial_interface import (  # noqa: PLC0415
            SerialInterface,
        )
    except ImportError as e:
        msg = (
            "Could not import meshtastic python package. "
            "Please install it with `pip install meshtastic`."
        )
        raise ImportError(msg) from e
    return SerialInterface


class MeshtasticSendInput(BaseModel):
    """Input schema for the MeshtasticSendTool.

    Defines the parameters that an LLM can use when invoking the tool.
    """

    message: str = Field(
        ...,
        description="The text message to broadcast to the mesh network.",
    )
    channel_index: int = Field(
        default=0,
        description=(
            "The channel index to send the message on. Use 0 for the primary channel."
        ),
        ge=0,
        le=7,
    )


class MeshtasticSendTool(BaseTool):  # type: ignore[override]
    """Tool for sending messages to a Meshtastic LoRa mesh network.

    This tool enables LangChain agents to communicate via Meshtastic devices,
    which use LoRa (Long Range) radio technology to create decentralized mesh
    networks. This is particularly useful for:

    - Offline-first AI applications where internet connectivity is unavailable
    - Disaster recovery scenarios where traditional infrastructure is down
    - Remote area communications beyond cellular coverage
    - Privacy-focused messaging that doesn't rely on centralized servers

    Setup:
        Install ``langchain-meshtastic`` and the ``meshtastic`` package:

        .. code-block:: bash

            pip install langchain-meshtastic meshtastic

        Connect a Meshtastic-compatible device via USB.

    Instantiation:
        .. code-block:: python

            from langchain_meshtastic import MeshtasticSendTool

            # Auto-detect connected device
            tool = MeshtasticSendTool()

            # Or specify device path explicitly
            tool = MeshtasticSendTool(device_path="/dev/ttyUSB0")

    Invocation with args:
        .. code-block:: python

            tool.invoke({"message": "Hello from AI!", "channel_index": 0})

        .. code-block:: python

            "Successfully sent message to mesh network on channel 0: 'Hello from AI!'"

    Invocation with ToolCall:
        .. code-block:: python

            tool.invoke(
                {
                    "args": {"message": "Emergency alert!", "channel_index": 1},
                    "id": "1",
                    "name": tool.name,
                    "type": "tool_call",
                }
            )

        .. code-block:: python

            ToolMessage(
                content="Successfully sent message to mesh network on channel 1: 'Emergency alert!'",
                name="meshtastic_send",
                tool_call_id="1",
            )
    """  # noqa: E501

    name: str = "meshtastic_send"
    """The unique name of the tool."""

    description: str = (
        "Send a text message to a Meshtastic LoRa mesh network. "
        "Use this to communicate in offline environments or areas without "
        "cellular/internet coverage. Requires a connected Meshtastic device."
    )
    """Description shown to the LLM to help it decide when to use this tool."""

    args_schema: type[BaseModel] = MeshtasticSendInput
    """Pydantic model defining the input schema for this tool."""

    device_path: str | None = None
    """Optional serial device path (e.g., '/dev/ttyUSB0' or 'COM3').

    If not specified, the tool will attempt to auto-detect the connected device.
    """

    def _run(
        self,
        message: str,
        channel_index: int = 0,
        run_manager: CallbackManagerForToolRun | None = None,  # noqa: ARG002
    ) -> str:
        """Send a message to the Meshtastic mesh network.

        Args:
            message: The text message to broadcast.
            channel_index: The channel index to send on (0-7).
            run_manager: Optional callback manager for tracing.

        Returns:
            A status message indicating success or describing the failure.
        """
        try:
            serial_interface_cls = _get_meshtastic_serial_interface()
        except ImportError as e:
            return str(e)

        interface: SerialInterface | None = None
        try:
            # Initialize connection to the Meshtastic device
            interface = serial_interface_cls(devPath=self.device_path)

            # Send the message to the mesh network
            interface.sendText(
                text=message,
                channelIndex=channel_index,
                wantAck=True,
            )
        except FileNotFoundError:
            return (
                "Error: No Meshtastic device found. "
                "Please ensure a device is connected via USB."
            )
        except PermissionError:
            return (
                "Error: Permission denied when accessing Meshtastic device. "
                "You may need to add your user to the 'dialout' group or run with "
                "elevated permissions."
            )
        except (OSError, RuntimeError) as e:
            return f"Error sending message via Meshtastic: {e}"
        else:
            return (
                f"Successfully sent message to mesh network on channel "
                f"{channel_index}: '{message}'"
            )
        finally:
            if interface is not None:
                with contextlib.suppress(Exception):
                    interface.close()
