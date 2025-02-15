import threading
from typing import Any, Dict, List, Union, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from langchain_community.callbacks.bedrock_anthropic_callback import BedrockAnthropicTokenUsageCallbackHandler

MODEL_COST_PER_1K_INPUT_TOKENS = {
    "amazon.nova-micro-v1:0": 0.000035,
    "amazon.nova-lite-v1:0": 0.00006,
    "amazon.nova-pro-v1:0": 0.0008,
}

MODEL_COST_PER_1K_OUTPUT_TOKENS = {
    "amazon.nova-micro-v1:0": 0.00014,
    "amazon.nova-lite-v1:0": 0.00024,
    "amazon.nova-pro-v1:0": 0.0032,
}

def _get_token_cost(
        prompt_tokens: int, completion_tokens: int, model_id: Union[str, None]) -> float:
    if model_id:
        # The model ID can be a cross-region (system-defined) inference profile ID,
        # which has a prefix indicating the region (e.g., 'us', 'eu') but
        # shares the same token costs as the "base model".
        # By extracting the "base model ID", by taking the last two segments
        # of the model ID, we can map cross-region inference profile IDs to
        # their corresponding cost entries.
        base_model_id = model_id.split(".")[-2] + "." + model_id.split(".")[-1]
    else:
        base_model_id = None
    print("#### "+str(base_model_id))
    """Get the cost of tokens for the model."""
    if base_model_id not in MODEL_COST_PER_1K_INPUT_TOKENS:
        raise ValueError(
            f"Unknown model: {model_id}. Please provide a valid  model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_INPUT_TOKENS.keys())
        )
    return (prompt_tokens / 1000) * MODEL_COST_PER_1K_INPUT_TOKENS[base_model_id] + (
        completion_tokens / 1000
    ) * MODEL_COST_PER_1K_OUTPUT_TOKENS[base_model_id]

class BedrockTokenUsageCallbackHandler(BedrockAnthropicTokenUsageCallbackHandler):
    """Generic Bedrock callback handler that supports multiple model providers."""

    model_id: Optional[str] = None
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        # Remember model_id here, because it is not available in on_llm_end
        self.model_id = kwargs["run_id"]["invocation_params"]["model_id"]

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if response.llm_output:
            super().on_llm_end(response, **kwargs)
            return
        if response.generations is None:
            return None

        completion_tokens = 0
        prompt_tokens = 0
        total_tokens = 0
        for gen in response.generations:
            for chatGeneration in gen:
                jsondata = chatGeneration.to_json()
                if "kwargs" not in jsondata:
                    continue
                if "message" not in jsondata["kwargs"]:
                    continue
                usage_metadata = jsondata["kwargs"]["message"].usage_metadata
                completion_tokens += usage_metadata["input_tokens"]
                prompt_tokens += usage_metadata["output_tokens"]
                total_tokens += usage_metadata["input_tokens"] + usage_metadata["output_tokens"]
        
        total_cost = _get_token_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_id=self.model_id,
        )

        

        # update shared state behind lock
        
        with self._lock:
            self.total_cost += total_cost
            self.total_tokens += total_tokens
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.successful_requests += 1

    def __copy__(self) -> "BedrockTokenUsageCallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "BedrockTokenUsageCallbackHandler":
        """Return a deep copy of the callback handler."""
        return self
    