from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM


from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


class VertexAIMedLM(LLM):
    """Adapted  from https://python.langchain.com/v0.1/docs/modules/model_io/llms/custom_llm/"""
    parameters_dict: dict = {
        "candidateCount": 1,
        "maxOutputTokens": 1024,
        "temperature": 0,
        "topP": 0.8,
        "topK": 40
    }
    model_name: str = "medlm-large"
    gcp_project_id: str

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """

        client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
        client_obj = aiplatform.gapic.PredictionServiceClient(
            client_options=client_options
        )
        parameters = json_format.ParseDict(self.parameters_dict, Value())

        instance_dict = {
            "content": prompt
        }
        instance = json_format.ParseDict(instance_dict, Value())
        instances = [instance]

        response =  client_obj.predict(
            endpoint=f"projects/{self.gcp_project_id}/locations/us-central1/publishers/google/models/{self.model_name}",
            instances=instances, parameters=parameters
        )
        print("response")
        predictions = response.predictions
        if predictions and len(predictions) > 0:
            return predictions[0]['content']
        return "No predictions found"


    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "VertexAI-MedLM-Wrapper"


if __name__ == "__main__":
    llm = VertexAIMedLM(model_name="medlm-large", gcp_project_id="my-gcp-project")
    print(llm.invoke("hello"))