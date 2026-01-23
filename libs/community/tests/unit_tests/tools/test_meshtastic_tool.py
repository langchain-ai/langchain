"""Unit tests for MeshtasticSendTool."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_serial_interface():
    """Create a mock SerialInterface for testing without hardware."""
    mock_interface = MagicMock()
    mock_interface.sendText = MagicMock()
    mock_interface.close = MagicMock()
    return mock_interface


@pytest.fixture
def mock_meshtastic(mock_serial_interface):
    """Mock the meshtastic import and SerialInterface class."""
    with patch(
        "langchain_community.tools.meshtastic_tool._get_meshtastic_serial_interface"
    ) as mock_get:
        mock_get.return_value = MagicMock(return_value=mock_serial_interface)
        yield mock_serial_interface


class TestMeshtasticSendTool:
    """Tests for MeshtasticSendTool."""

    def test_tool_name_and_description(self):
        """Test that the tool has correct name and description."""
        from langchain_community.tools.meshtastic_tool import MeshtasticSendTool

        tool = MeshtasticSendTool()

        assert tool.name == "meshtastic_send"
        assert "Meshtastic" in tool.description
        assert "LoRa" in tool.description

    def test_tool_args_schema(self):
        """Test that args_schema is properly configured."""
        from langchain_community.tools.meshtastic_tool import (
            MeshtasticSendInput,
            MeshtasticSendTool,
        )

        tool = MeshtasticSendTool()

        assert tool.args_schema == MeshtasticSendInput
        # Verify schema has expected fields
        schema = tool.args_schema.model_json_schema()
        assert "message" in schema["properties"]
        assert "channel_index" in schema["properties"]

    def test_input_schema_validation(self):
        """Test input schema validation."""
        from langchain_community.tools.meshtastic_tool import MeshtasticSendInput

        # Valid input
        valid_input = MeshtasticSendInput(message="Hello", channel_index=0)
        assert valid_input.message == "Hello"
        assert valid_input.channel_index == 0

        # Default channel_index
        default_input = MeshtasticSendInput(message="Test")
        assert default_input.channel_index == 0

        # Invalid channel_index (out of range)
        with pytest.raises(ValueError):
            MeshtasticSendInput(message="Test", channel_index=10)

        with pytest.raises(ValueError):
            MeshtasticSendInput(message="Test", channel_index=-1)

    def test_send_message_success(self, mock_meshtastic):
        """Test successful message sending."""
        from langchain_community.tools.meshtastic_tool import MeshtasticSendTool

        tool = MeshtasticSendTool()
        result = tool._run(message="Hello World", channel_index=1)

        # Verify sendText was called with correct arguments
        mock_meshtastic.sendText.assert_called_once_with(
            text="Hello World",
            channelIndex=1,
            wantAck=True,
        )

        # Verify interface was closed
        mock_meshtastic.close.assert_called_once()

        # Verify success message
        assert "Successfully sent" in result
        assert "Hello World" in result
        assert "channel 1" in result

    def test_send_message_default_channel(self, mock_meshtastic):
        """Test sending with default channel index."""
        from langchain_community.tools.meshtastic_tool import MeshtasticSendTool

        tool = MeshtasticSendTool()
        tool._run(message="Test message")

        mock_meshtastic.sendText.assert_called_once_with(
            text="Test message",
            channelIndex=0,
            wantAck=True,
        )

    def test_send_message_with_device_path(self, mock_serial_interface):
        """Test that device_path is passed to SerialInterface."""
        from langchain_community.tools.meshtastic_tool import MeshtasticSendTool

        mock_class = MagicMock(return_value=mock_serial_interface)

        with patch(
            "langchain_community.tools.meshtastic_tool._get_meshtastic_serial_interface",
            return_value=mock_class,
        ):
            tool = MeshtasticSendTool(device_path="/dev/ttyUSB0")
            tool._run(message="Test")

            mock_class.assert_called_once_with(devPath="/dev/ttyUSB0")

    def test_missing_meshtastic_library(self):
        """Test error handling when meshtastic library is not installed."""
        from langchain_community.tools.meshtastic_tool import MeshtasticSendTool

        with patch(
            "langchain_community.tools.meshtastic_tool._get_meshtastic_serial_interface",
            side_effect=ImportError("Could not import meshtastic"),
        ):
            tool = MeshtasticSendTool()
            result = tool._run(message="Test")

            assert "meshtastic" in result.lower()

    def test_device_not_found(self, mock_serial_interface):
        """Test error handling when device is not connected."""
        from langchain_community.tools.meshtastic_tool import MeshtasticSendTool

        mock_class = MagicMock(side_effect=FileNotFoundError("Device not found"))

        with patch(
            "langchain_community.tools.meshtastic_tool._get_meshtastic_serial_interface",
            return_value=mock_class,
        ):
            tool = MeshtasticSendTool()
            result = tool._run(message="Test")

            assert "No Meshtastic device found" in result

    def test_permission_denied(self, mock_serial_interface):
        """Test error handling for permission issues."""
        from langchain_community.tools.meshtastic_tool import MeshtasticSendTool

        mock_class = MagicMock(side_effect=PermissionError("Access denied"))

        with patch(
            "langchain_community.tools.meshtastic_tool._get_meshtastic_serial_interface",
            return_value=mock_class,
        ):
            tool = MeshtasticSendTool()
            result = tool._run(message="Test")

            assert "Permission denied" in result

    def test_generic_exception_handling(self, mock_meshtastic):
        """Test handling of unexpected exceptions."""
        from langchain_community.tools.meshtastic_tool import MeshtasticSendTool

        mock_meshtastic.sendText.side_effect = Exception("Unexpected error")

        tool = MeshtasticSendTool()
        result = tool._run(message="Test")

        assert "Error sending message" in result
        assert "Unexpected error" in result

        # Ensure close is still called even after exception
        mock_meshtastic.close.assert_called_once()

    def test_interface_closed_on_success(self, mock_meshtastic):
        """Test that interface is always closed after successful send."""
        from langchain_community.tools.meshtastic_tool import MeshtasticSendTool

        tool = MeshtasticSendTool()
        tool._run(message="Test")

        mock_meshtastic.close.assert_called_once()

    def test_interface_closed_on_failure(self, mock_meshtastic):
        """Test that interface is closed even when send fails."""
        from langchain_community.tools.meshtastic_tool import MeshtasticSendTool

        mock_meshtastic.sendText.side_effect = RuntimeError("Send failed")

        tool = MeshtasticSendTool()
        tool._run(message="Test")

        mock_meshtastic.close.assert_called_once()


class TestMeshtasticSendInput:
    """Tests for MeshtasticSendInput schema."""

    def test_required_message(self):
        """Test that message is required."""
        from langchain_community.tools.meshtastic_tool import MeshtasticSendInput

        with pytest.raises(ValueError):
            MeshtasticSendInput()

    def test_channel_index_bounds(self):
        """Test channel_index is constrained to valid range."""
        from langchain_community.tools.meshtastic_tool import MeshtasticSendInput

        # Valid channel indices (0-7)
        for i in range(8):
            input_model = MeshtasticSendInput(message="Test", channel_index=i)
            assert input_model.channel_index == i

        # Invalid channel indices
        with pytest.raises(ValueError):
            MeshtasticSendInput(message="Test", channel_index=8)

        with pytest.raises(ValueError):
            MeshtasticSendInput(message="Test", channel_index=-1)
