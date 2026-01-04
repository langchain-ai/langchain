"""Unit tests for QWED tool."""
import pytest
from unittest.mock import Mock, patch
from langchain_community.tools.qwed import QWEDTool


def test_qwed_tool_initialization():
    """Test QWED tool can be initialized with defaults."""
    tool = QWEDTool()
    assert tool.name == "qwed_verify"
    assert "neurosymbolic" in tool.description.lower()
    assert tool.provider == "openai"
    assert tool.model == "gpt-4o-mini"
    assert tool.mask_pii is False


def test_qwed_tool_custom_config():
    """Test QWED tool can be initialized with custom config."""
    tool = QWEDTool(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        mask_pii=True
    )
    assert tool.provider == "anthropic"
    assert tool.model == "claude-3-haiku-20240307"
    assert tool.mask_pii is True


def test_qwed_tool_schema():
    """Test input schema is correctly defined."""
    tool = QWEDTool()
    schema = tool.args_schema.schema()
    
    assert "query" in schema["properties"]
    assert schema["properties"]["query"]["type"] == "string"
    assert "derivative" in schema["properties"]["query"]["description"]


@pytest.mark.skipif(
    not _has_qwed(),
    reason="QWED not installed"
)
def test_qwed_tool_math_verification():
    """Test math verification with mocked QWED."""
    tool = QWEDTool(provider="openai")
    
    # Mock the QWED client
    with patch("langchain_community.tools.qwed.tool.QWEDLocal") as mock_qwed:
        # Create mock result
        mock_result = Mock()
        mock_result.verified = True
        mock_result.value = "2*x"
        mock_result.confidence = 1.0
        mock_result.evidence = {"method": "symbolic"}
        mock_result.error = None
        
        # Set up mock
        mock_instance = Mock()
        mock_instance.verify.return_value = mock_result
        mock_qwed.return_value = mock_instance
        
        # Run tool
        result = tool._run("What is the derivative of x^2?")
        
        # Verify
        assert "✅ VERIFIED" in result
        assert "2*x" in result
        assert "100%" in result


@pytest.mark.skipif(
    not _has_qwed(),
    reason="QWED not installed"
)
def test_qwed_tool_verification_failure():
    """Test verification failure handling."""
    tool = QWEDTool()
    
    with patch("langchain_community.tools.qwed.tool.QWEDLocal") as mock_qwed:
        # Create mock failure
        mock_result = Mock()
        mock_result.verified = False
        mock_result.value = None
        mock_result.confidence = 0.0
        mock_result.error = "Could not parse mathematical expression"
        mock_result.evidence = {}
        
        mock_instance = Mock()
        mock_instance.verify.return_value = mock_result
        mock_qwed.return_value = mock_instance
        
        result = tool._run("invalid query")
        
        assert "❌ VERIFICATION FAILED" in result
        assert "Could not parse" in result


def test_qwed_tool_import_error():
    """Test graceful handling when QWED not installed."""
    tool = QWEDTool()
    
    with patch.dict("sys.modules", {"qwed_sdk": None}):
        with pytest.raises(ImportError, match="QWED is not installed"):
            tool._run("test query")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _has_qwed(),
    reason="QWED not installed"
)
async def test_qwed_tool_async():
    """Test async version delegates to sync."""
    tool = QWEDTool()
    
    with patch("langchain_community.tools.qwed.tool.QWEDLocal") as mock_qwed:
        mock_result = Mock()
        mock_result.verified = True
        mock_result.value = "4"
        mock_result.confidence = 1.0
        mock_result.evidence = {"method": "symbolic"}
        mock_result.error = None
        
        mock_instance = Mock()
        mock_instance.verify.return_value = mock_result
        mock_qwed.return_value = mock_instance
        
        result = await tool._arun("What is 2+2?")
        
        assert "✅ VERIFIED" in result
        assert "4" in result


def _has_qwed() -> bool:
    """Check if QWED is installed."""
    try:
        import qwed_sdk
        return True
    except ImportError:
        return False
