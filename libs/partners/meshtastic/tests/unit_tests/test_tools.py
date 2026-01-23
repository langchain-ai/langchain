"""Unit tests for MeshtasticSendTool."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from langchain_meshtastic import MeshtasticSendInput, MeshtasticSendTool


@pytest.fixture
def mock_serial_interface() -> MagicMock:
    """Create a mock SerialInterface for testing without hardware."""
    mock_interface = MagicMock()
    mock_interface.sendText = MagicMock()
    mock_interface.close = MagicMock()
    return mock_interface


@pytest.fixture
def mock_meshtastic(
    mock_serial_interface: MagicMock,
) -> Generator[MagicMock, None, None]:
    """Mock the meshtastic import and SerialInterface class."""
    with patch(
        "langchain_meshtastic.tools._get_meshtastic_serial_interface"
    ) as mock_get:
        mock_get.return_value = MagicMock(return_value=mock_serial_interface)
        yield mock_serial_interface


class TestMeshtasticSendTool:
    """Tests for MeshtasticSendTool."""

    def test_tool_name_and_description(self) -> None:
        """Test that the tool has correct name and description."""
        tool = MeshtasticSendTool()

        assert tool.name == "meshtastic_send"
        assert "Meshtastic" in tool.description
        assert "LoRa" in tool.description

    def test_tool_args_schema(self) -> None:
        """Test that args_schema is properly configured."""
        tool = MeshtasticSendTool()

        assert tool.args_schema == MeshtasticSendInput
        schema = tool.args_schema.model_json_schema()
        assert "message" in schema["properties"]
        assert "channel_index" in schema["properties"]

    def test_send_message_success(self, mock_meshtastic: MagicMock) -> None:
        """Test successful message sending."""
        tool = MeshtasticSendTool()
        result = tool._run(message="Hello World", channel_index=1)

        mock_meshtastic.sendText.assert_called_once_with(
            text="Hello World",
            channelIndex=1,
            wantAck=True,
        )
        mock_meshtastic.close.assert_called_once()
        assert "Successfully sent" in result
        assert "Hello World" in result
        assert "channel 1" in result

    def test_send_message_default_channel(self, mock_meshtastic: MagicMock) -> None:
        """Test sending with default channel index."""
        tool = MeshtasticSendTool()
        tool._run(message="Test message")

        mock_meshtastic.sendText.assert_called_once_with(
            text="Test message",
            channelIndex=0,
            wantAck=True,
        )

    def test_send_message_with_device_path(
        self, mock_serial_interface: MagicMock
    ) -> None:
        """Test that device_path is passed to SerialInterface."""
        mock_class = MagicMock(return_value=mock_serial_interface)

        with patch(
            "langchain_meshtastic.tools._get_meshtastic_serial_interface",
            return_value=mock_class,
        ):
            tool = MeshtasticSendTool(device_path="/dev/ttyUSB0")
            tool._run(message="Test")

            mock_class.assert_called_once_with(devPath="/dev/ttyUSB0")

    def test_missing_meshtastic_library(self) -> None:
        """Test error handling when meshtastic library is not installed."""
        with patch(
            "langchain_meshtastic.tools._get_meshtastic_serial_interface",
            side_effect=ImportError("Could not import meshtastic"),
        ):
            tool = MeshtasticSendTool()
            result = tool._run(message="Test")

            assert "meshtastic" in result.lower()

    def test_device_not_found(self, mock_serial_interface: MagicMock) -> None:
        """Test error handling when device is not connected."""
        mock_class = MagicMock(side_effect=FileNotFoundError("Device not found"))

        with patch(
            "langchain_meshtastic.tools._get_meshtastic_serial_interface",
            return_value=mock_class,
        ):
            tool = MeshtasticSendTool()
            result = tool._run(message="Test")

            assert "No Meshtastic device found" in result

    def test_permission_denied(self, mock_serial_interface: MagicMock) -> None:
        """Test error handling for permission issues."""
        mock_class = MagicMock(side_effect=PermissionError("Access denied"))

        with patch(
            "langchain_meshtastic.tools._get_meshtastic_serial_interface",
            return_value=mock_class,
        ):
            tool = MeshtasticSendTool()
            result = tool._run(message="Test")

            assert "Permission denied" in result

    def test_generic_exception_handling(self, mock_meshtastic: MagicMock) -> None:
        """Test handling of unexpected exceptions."""
        mock_meshtastic.sendText.side_effect = RuntimeError("Unexpected error")

        tool = MeshtasticSendTool()
        result = tool._run(message="Test")

        assert "Error sending message" in result
        assert "Unexpected error" in result
        mock_meshtastic.close.assert_called_once()

    def test_interface_closed_on_success(self, mock_meshtastic: MagicMock) -> None:
        """Test that interface is always closed after successful send."""
        tool = MeshtasticSendTool()
        tool._run(message="Test")

        mock_meshtastic.close.assert_called_once()

    def test_interface_closed_on_failure(self, mock_meshtastic: MagicMock) -> None:
        """Test that interface is closed even when send fails."""
        mock_meshtastic.sendText.side_effect = OSError("Send failed")

        tool = MeshtasticSendTool()
        tool._run(message="Test")

        mock_meshtastic.close.assert_called_once()


class TestMeshtasticSendInput:
    """Tests for MeshtasticSendInput schema."""

    def test_valid_input(self) -> None:
        """Test valid input creation."""
        input_model = MeshtasticSendInput(message="Hello", channel_index=0)
        assert input_model.message == "Hello"
        assert input_model.channel_index == 0

    def test_default_channel_index(self) -> None:
        """Test that channel_index defaults to 0."""
        input_model = MeshtasticSendInput(message="Test")
        assert input_model.channel_index == 0

    def test_channel_index_bounds(self) -> None:
        """Test channel_index is constrained to valid range."""
        for i in range(8):
            input_model = MeshtasticSendInput(message="Test", channel_index=i)
            assert input_model.channel_index == i

        with pytest.raises(ValueError):
            MeshtasticSendInput(message="Test", channel_index=8)

        with pytest.raises(ValueError):
            MeshtasticSendInput(message="Test", channel_index=-1)

    def test_required_message(self) -> None:
        """Test that message is required."""
        with pytest.raises(ValueError):
            MeshtasticSendInput()  # type: ignore[call-arg]
