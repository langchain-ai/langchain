"""Test ModelsLab image generation tools."""
import pytest
from unittest.mock import Mock, patch

from langchain_community.tools.modelslab import (
    ModelsLabImageGenerationTool,
    ModelsLabImageToImageTool,
    ModelsLabInpaintingTool,
    ModelsLabMultiImageTool,
)
from langchain_community.utilities.modelslab import ModelsLabAPIWrapper


class TestModelsLabImageGenerationTool:
    """Test ModelsLabImageGenerationTool functionality."""

    @pytest.fixture
    def mock_api_wrapper(self):
        """Create a mock API wrapper."""
        wrapper = Mock(spec=ModelsLabAPIWrapper)
        wrapper.run.return_value = "https://example.com/generated-image.jpg"
        return wrapper

    @pytest.fixture
    def image_tool(self, mock_api_wrapper):
        """Create a ModelsLabImageGenerationTool instance."""
        return ModelsLabImageGenerationTool(api_wrapper=mock_api_wrapper)

    def test_tool_initialization(self, mock_api_wrapper):
        """Test tool initialization."""
        tool = ModelsLabImageGenerationTool(api_wrapper=mock_api_wrapper)
        
        assert tool.name == "modelslab_image_generation"
        assert "Generate high-quality images" in tool.description
        assert tool.api_wrapper == mock_api_wrapper

    def test_tool_run_success(self, image_tool, mock_api_wrapper):
        """Test successful tool execution."""
        prompt = "A beautiful mountain landscape at sunset"
        result = image_tool.run(prompt)
        
        assert result == "https://example.com/generated-image.jpg"
        mock_api_wrapper.run.assert_called_once_with(prompt)

    def test_tool_run_with_callback_manager(self, image_tool, mock_api_wrapper):
        """Test tool execution with callback manager."""
        mock_callback_manager = Mock()
        
        prompt = "A serene lake reflection"
        result = image_tool._run(prompt, run_manager=mock_callback_manager)
        
        assert result == "https://example.com/generated-image.jpg"
        mock_api_wrapper.run.assert_called_once_with(prompt)
        
        # Verify callback manager was used
        mock_callback_manager.on_tool_start.assert_called_once()
        mock_callback_manager.on_tool_end.assert_called_once_with(result)

    def test_tool_run_error_handling(self, image_tool, mock_api_wrapper):
        """Test tool error handling."""
        mock_api_wrapper.run.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="ModelsLab image generation failed: API Error"):
            image_tool.run("Test prompt")

    def test_tool_run_error_with_callback(self, image_tool, mock_api_wrapper):
        """Test tool error handling with callback manager."""
        mock_callback_manager = Mock()
        mock_api_wrapper.run.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            image_tool._run("Test prompt", run_manager=mock_callback_manager)
            
        mock_callback_manager.on_tool_error.assert_called_once()

    def test_tool_input_schema(self, image_tool):
        """Test tool input schema validation."""
        schema = image_tool.args_schema.model_json_schema()
        
        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"
        assert "detailed text description" in schema["properties"]["query"]["description"]

    def test_tool_description_content(self, image_tool):
        """Test that tool description contains key information."""
        description = image_tool.description
        
        # Check for key features mentioned
        assert "ModelsLab" in description
        assert "Flux" in description
        assert "Stable Diffusion" in description
        assert "text prompt" in description
        assert "artwork" in description or "illustrations" in description


class TestModelsLabImageToImageTool:
    """Test ModelsLabImageToImageTool functionality."""

    @pytest.fixture
    def mock_api_wrapper(self):
        """Create a mock API wrapper."""
        wrapper = Mock(spec=ModelsLabAPIWrapper)
        wrapper.run.return_value = "https://example.com/transformed-image.jpg"
        return wrapper

    @pytest.fixture
    def img2img_tool(self, mock_api_wrapper):
        """Create a ModelsLabImageToImageTool instance."""
        return ModelsLabImageToImageTool(api_wrapper=mock_api_wrapper)

    def test_img2img_tool_initialization(self, mock_api_wrapper):
        """Test img2img tool initialization."""
        tool = ModelsLabImageToImageTool(api_wrapper=mock_api_wrapper)
        
        assert tool.name == "modelslab_img2img"
        assert "Transform existing images" in tool.description
        assert "style transfer" in tool.description

    def test_img2img_tool_run(self, img2img_tool, mock_api_wrapper):
        """Test img2img tool execution."""
        prompt = "Make this image cyberpunk style"
        result = img2img_tool.run(prompt)
        
        assert result == "https://example.com/transformed-image.jpg"
        mock_api_wrapper.run.assert_called_once_with(prompt)

    def test_img2img_tool_error_handling(self, img2img_tool, mock_api_wrapper):
        """Test img2img tool error handling."""
        mock_api_wrapper.run.side_effect = Exception("Transform failed")
        
        with pytest.raises(Exception, match="ModelsLab img2img failed: Transform failed"):
            img2img_tool.run("Test prompt")


class TestModelsLabInpaintingTool:
    """Test ModelsLabInpaintingTool functionality."""

    @pytest.fixture
    def mock_api_wrapper(self):
        """Create a mock API wrapper."""
        wrapper = Mock(spec=ModelsLabAPIWrapper)
        wrapper.run.return_value = "https://example.com/inpainted-image.jpg"
        return wrapper

    @pytest.fixture
    def inpaint_tool(self, mock_api_wrapper):
        """Create a ModelsLabInpaintingTool instance."""
        return ModelsLabInpaintingTool(api_wrapper=mock_api_wrapper)

    def test_inpaint_tool_initialization(self, mock_api_wrapper):
        """Test inpainting tool initialization."""
        tool = ModelsLabInpaintingTool(api_wrapper=mock_api_wrapper)
        
        assert tool.name == "modelslab_inpainting"
        assert "Edit specific regions" in tool.description
        assert "masks" in tool.description

    def test_inpaint_tool_run(self, inpaint_tool, mock_api_wrapper):
        """Test inpainting tool execution."""
        prompt = "A red sports car in this location"
        result = inpaint_tool.run(prompt)
        
        assert result == "https://example.com/inpainted-image.jpg"
        mock_api_wrapper.run.assert_called_once_with(prompt)

    def test_inpaint_tool_error_handling(self, inpaint_tool, mock_api_wrapper):
        """Test inpainting tool error handling."""
        mock_api_wrapper.run.side_effect = Exception("Inpaint failed")
        
        with pytest.raises(Exception, match="ModelsLab inpainting failed: Inpaint failed"):
            inpaint_tool.run("Test prompt")


class TestModelsLabMultiImageTool:
    """Test ModelsLabMultiImageTool functionality."""

    @pytest.fixture
    def mock_api_wrapper(self):
        """Create a mock API wrapper."""
        wrapper = Mock(spec=ModelsLabAPIWrapper)
        wrapper.run_multiple.return_value = [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
            "https://example.com/image3.jpg",
            "https://example.com/image4.jpg"
        ]
        return wrapper

    @pytest.fixture
    def multi_tool(self, mock_api_wrapper):
        """Create a ModelsLabMultiImageTool instance."""
        return ModelsLabMultiImageTool(api_wrapper=mock_api_wrapper)

    def test_multi_tool_initialization(self, mock_api_wrapper):
        """Test multi-image tool initialization."""
        tool = ModelsLabMultiImageTool(api_wrapper=mock_api_wrapper)
        
        assert tool.name == "modelslab_multi_image"
        assert "multiple image variations" in tool.description
        assert "2-4 different versions" in tool.description

    def test_multi_tool_run(self, multi_tool, mock_api_wrapper):
        """Test multi-image tool execution."""
        prompt = "A fantasy castle on a hill"
        result = multi_tool.run(prompt)
        
        expected_result = (
            "https://example.com/image1.jpg, "
            "https://example.com/image2.jpg, "
            "https://example.com/image3.jpg, "
            "https://example.com/image4.jpg"
        )
        
        assert result == expected_result
        mock_api_wrapper.run_multiple.assert_called_once_with(prompt, samples=4)

    def test_multi_tool_error_handling(self, multi_tool, mock_api_wrapper):
        """Test multi-image tool error handling."""
        mock_api_wrapper.run_multiple.side_effect = Exception("Multi-gen failed")
        
        with pytest.raises(Exception, match="ModelsLab multi-image generation failed: Multi-gen failed"):
            multi_tool.run("Test prompt")

    def test_multi_tool_input_schema(self, multi_tool):
        """Test multi-image tool input schema."""
        schema = multi_tool.args_schema.model_json_schema()
        
        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"


class TestToolIntegration:
    """Test integration scenarios with multiple tools."""

    def test_all_tools_have_unique_names(self):
        """Test that all tools have unique names."""
        with patch.dict("os.environ", {"MODELSLAB_API_KEY": "test-key"}):
            api_wrapper = ModelsLabAPIWrapper()
            
            tools = [
                ModelsLabImageGenerationTool(api_wrapper=api_wrapper),
                ModelsLabImageToImageTool(api_wrapper=api_wrapper),
                ModelsLabInpaintingTool(api_wrapper=api_wrapper),
                ModelsLabMultiImageTool(api_wrapper=api_wrapper),
            ]
            
            tool_names = [tool.name for tool in tools]
            assert len(tool_names) == len(set(tool_names))  # All names are unique

    def test_all_tools_have_descriptions(self):
        """Test that all tools have meaningful descriptions."""
        with patch.dict("os.environ", {"MODELSLAB_API_KEY": "test-key"}):
            api_wrapper = ModelsLabAPIWrapper()
            
            tools = [
                ModelsLabImageGenerationTool(api_wrapper=api_wrapper),
                ModelsLabImageToImageTool(api_wrapper=api_wrapper),
                ModelsLabInpaintingTool(api_wrapper=api_wrapper),
                ModelsLabMultiImageTool(api_wrapper=api_wrapper),
            ]
            
            for tool in tools:
                assert len(tool.description) > 50  # Meaningful description
                assert "ModelsLab" in tool.description  # Brand mention
                assert tool.description.endswith(".")  # Proper sentence ending

    @patch.dict("os.environ", {"MODELSLAB_API_KEY": "test-key"})
    def test_tools_can_be_imported_and_instantiated(self):
        """Test that tools can be imported and instantiated properly."""
        from langchain_community.tools.modelslab import (
            ModelsLabImageGenerationTool,
            ModelsLabImageToImageTool,
            ModelsLabInpaintingTool,
            ModelsLabMultiImageTool,
        )
        from langchain_community.utilities.modelslab import ModelsLabAPIWrapper
        
        api_wrapper = ModelsLabAPIWrapper()
        
        # All tools should instantiate without error
        tools = [
            ModelsLabImageGenerationTool(api_wrapper=api_wrapper),
            ModelsLabImageToImageTool(api_wrapper=api_wrapper),
            ModelsLabInpaintingTool(api_wrapper=api_wrapper),
            ModelsLabMultiImageTool(api_wrapper=api_wrapper),
        ]
        
        assert len(tools) == 4
        for tool in tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'api_wrapper')
            assert callable(tool.run)