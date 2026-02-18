"""Test ModelsLabAPIWrapper utility."""
import pytest
from unittest.mock import Mock, patch
from langchain_community.utilities.modelslab import ModelsLabAPIWrapper


class TestModelsLabAPIWrapper:
    """Test ModelsLabAPIWrapper functionality."""

    @pytest.fixture
    def mock_api_wrapper(self):
        """Create a mock ModelsLab API wrapper for testing."""
        with patch.dict("os.environ", {"MODELSLAB_API_KEY": "test-key"}):
            return ModelsLabAPIWrapper()

    def test_initialization_with_api_key(self):
        """Test wrapper initialization with API key."""
        wrapper = ModelsLabAPIWrapper(modelslab_api_key="test-key")
        assert wrapper.modelslab_api_key.get_secret_value() == "test-key"
        assert wrapper.model == "flux"
        assert wrapper.size == "1024x1024"
        assert wrapper.samples == 1

    def test_initialization_from_env(self):
        """Test wrapper initialization from environment variable."""
        with patch.dict("os.environ", {"MODELSLAB_API_KEY": "env-key"}):
            wrapper = ModelsLabAPIWrapper()
            assert wrapper.modelslab_api_key.get_secret_value() == "env-key"

    def test_request_payload_basic(self, mock_api_wrapper):
        """Test basic request payload generation."""
        payload = mock_api_wrapper._get_request_payload("A beautiful sunset")
        
        expected_keys = [
            "key", "model_id", "prompt", "width", "height", 
            "samples", "num_inference_steps", "guidance_scale",
            "safety_checker", "enhance_prompt"
        ]
        
        for key in expected_keys:
            assert key in payload
            
        assert payload["prompt"] == "A beautiful sunset"
        assert payload["width"] == 1024
        assert payload["height"] == 1024
        assert payload["model_id"] == "flux"

    def test_request_payload_with_overrides(self, mock_api_wrapper):
        """Test request payload with parameter overrides."""
        payload = mock_api_wrapper._get_request_payload(
            "Test prompt",
            model_id="stable-diffusion-xl",
            samples=2,
            negative_prompt="blurry",
            seed=123
        )
        
        assert payload["model_id"] == "stable-diffusion-xl"
        assert payload["samples"] == 2
        assert payload["negative_prompt"] == "blurry"
        assert payload["seed"] == 123

    def test_request_payload_img2img(self, mock_api_wrapper):
        """Test request payload for image-to-image generation."""
        payload = mock_api_wrapper._get_request_payload(
            "Transform this image",
            init_image="base64-image-data",
            strength=0.8
        )
        
        assert payload["init_image"] == "base64-image-data"
        assert payload["strength"] == 0.8

    def test_request_payload_inpaint(self, mock_api_wrapper):
        """Test request payload for inpainting."""
        payload = mock_api_wrapper._get_request_payload(
            "Paint a car here",
            init_image="base64-image-data",
            mask_image="base64-mask-data"
        )
        
        assert payload["init_image"] == "base64-image-data"
        assert payload["mask_image"] == "base64-mask-data"

    @patch('requests.post')
    def test_successful_generation(self, mock_post, mock_api_wrapper):
        """Test successful image generation."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "success",
            "output": ["https://example.com/image.jpg"]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = mock_api_wrapper.run("A beautiful landscape")
        
        assert result == "https://example.com/image.jpg"
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_successful_generation_base64(self, mock_post, mock_api_wrapper):
        """Test successful generation with base64 response."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "success", 
            "output": ["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = mock_api_wrapper.run("Test image")
        
        assert result.startswith("data:image/png;base64,")
        assert "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" in result

    @patch('requests.post')
    @patch('requests.get')
    @patch('time.sleep')
    def test_async_generation(self, mock_sleep, mock_get, mock_post, mock_api_wrapper):
        """Test async generation with polling."""
        # Mock initial async response
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "status": "processing",
            "id": "test-request-id"
        }
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response
        
        # Mock polling responses - first processing, then success
        mock_get_response_processing = Mock()
        mock_get_response_processing.json.return_value = {"status": "processing"}
        mock_get_response_processing.raise_for_status.return_value = None
        
        mock_get_response_success = Mock()
        mock_get_response_success.json.return_value = {
            "status": "success",
            "output": ["https://example.com/async-image.jpg"]
        }
        mock_get_response_success.raise_for_status.return_value = None
        
        mock_get.side_effect = [mock_get_response_processing, mock_get_response_success]

        result = mock_api_wrapper.run("Async generation test")
        
        assert result == "https://example.com/async-image.jpg"
        assert mock_get.call_count == 2
        mock_sleep.assert_called()

    @patch('requests.post')
    def test_multiple_images(self, mock_post, mock_api_wrapper):
        """Test multiple image generation."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "success",
            "output": [
                "https://example.com/image1.jpg",
                "https://example.com/image2.jpg",
                "https://example.com/image3.jpg"
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        results = mock_api_wrapper.run_multiple("Test prompt", samples=3)
        
        assert len(results) == 3
        assert results[0] == "https://example.com/image1.jpg"
        assert results[1] == "https://example.com/image2.jpg"
        assert results[2] == "https://example.com/image3.jpg"

    @patch('requests.post')
    def test_img2img_convenience_method(self, mock_post, mock_api_wrapper):
        """Test image-to-image convenience method."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "success",
            "output": ["https://example.com/transformed.jpg"]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = mock_api_wrapper.run_img2img(
            "Make this cyberpunk",
            "base64-image-data",
            strength=0.8
        )
        
        assert result == "https://example.com/transformed.jpg"
        
        # Verify the payload includes img2img parameters
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['init_image'] == "base64-image-data"
        assert payload['strength'] == 0.8

    @patch('requests.post')
    def test_inpaint_convenience_method(self, mock_post, mock_api_wrapper):
        """Test inpainting convenience method.""" 
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "success",
            "output": ["https://example.com/inpainted.jpg"]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = mock_api_wrapper.run_inpaint(
            "A red sports car",
            "base64-image-data",
            "base64-mask-data"
        )
        
        assert result == "https://example.com/inpainted.jpg"
        
        # Verify the payload includes inpainting parameters
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['init_image'] == "base64-image-data"
        assert payload['mask_image'] == "base64-mask-data"

    @patch('requests.post')
    def test_api_error_handling(self, mock_post, mock_api_wrapper):
        """Test API error handling."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError, match="ModelsLab API request failed"):
            mock_api_wrapper.run("Test prompt")

    @patch('requests.post')
    def test_empty_output_error(self, mock_post, mock_api_wrapper):
        """Test error handling for empty output."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "success",
            "output": []
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with pytest.raises(ValueError, match="No images generated"):
            mock_api_wrapper.run("Test prompt")

    @patch('requests.post')
    @patch('requests.get')
    def test_async_error_handling(self, mock_get, mock_post, mock_api_wrapper):
        """Test async operation error handling."""
        # Mock initial async response
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "status": "processing",
            "id": "test-request-id"
        }
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response
        
        # Mock error response from polling
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "status": "error",
            "message": "Generation failed"
        }
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        with pytest.raises(RuntimeError, match="ModelsLab generation failed: Generation failed"):
            mock_api_wrapper.run("Test prompt")

    def test_validation_samples_limit(self):
        """Test that samples are limited to maximum of 4."""
        wrapper = ModelsLabAPIWrapper(modelslab_api_key="test-key", samples=10)
        # Should be clamped to 4 in the field definition
        assert wrapper.samples == 4  # Field has le=4 constraint

    def test_validation_guidance_scale_range(self):
        """Test guidance scale validation."""
        # Valid range
        wrapper = ModelsLabAPIWrapper(modelslab_api_key="test-key", guidance_scale=7.5)
        assert wrapper.guidance_scale == 7.5
        
        # Should raise validation error for out of range values
        with pytest.raises(ValueError):
            ModelsLabAPIWrapper(modelslab_api_key="test-key", guidance_scale=25.0)