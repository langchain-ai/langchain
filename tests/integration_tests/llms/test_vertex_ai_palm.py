"""Test Google Vertex AI PaLM Text API wrapper.

   To use you must have the google-cloud-aiplatform Python package installed and
    either:

        1. Have credentials configured for your enviornment (gcloud, workload identity, etc...)
        2. Pass your service account key json using the google_application_credentials kwarg to the ChatGoogle
           constructor.

        *see: https://cloud.google.com/docs/authentication/application-default-credentials#GAC

"""

from pathlib import Path
import pytest

from langchain.llms.vertex_ai_palm import GoogleCloudVertexAIPalm
from langchain.llms.loading import load_llm

try:
    from vertexai.preview.language_models import TextGenerationModel # noqa: F401

    vertexai_installed = True
except ImportError:
    vertexai_installed = False

@pytest.mark.skipif(not vertexai_installed, reason="google-cloud-aiplatform>=1.25.0 package not installed")
def test_google_palm_call() -> None:
    """Test valid call to Google Vertex AI PaLM text API."""
    llm = GoogleCloudVertexAIPalm(max_output_tokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)

@pytest.mark.skipif(not vertexai_installed, reason="google-cloud-aiplatform>=1.25.0 package not installed")
def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading a Google PaLM LLM."""
    llm = GoogleCloudVertexAIPalm(max_output_tokens=10)
    llm.save(file_path=tmp_path / "google_palm.yaml")
    loaded_llm = load_llm(tmp_path / "google_palm.yaml")
    assert loaded_llm == llm
