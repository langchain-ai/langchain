import os
from typing import Any, List, Literal, Mapping, Optional

import genai

from langchain.llms.base import LLM


class WatsonxLLM(LLM):
    """
    BAM LLM connector class to langchain
    WIP: is lacking some functions
    """

    model_name: Literal[
        "salesforce/codegen-16b-mono",
        "prakharz/dial-flant5-xl",
        "tiiuae/falcon-40b",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
        "google/flan-ul2",
        "togethercomputer/gpt-jt-6b-v1",
        "eleutherai/gpt-neox-20b",
        "ibm/mpt-7b-instruct",
        "bigscience/mt0-xxl",
        "google/ul2",
    ] | str = "tiiuae/falcon-40b"
    api_key: str = os.environ["WATSONX_API_KEY"]
    decoding_method: str = "sample"
    temperature: float = 0.05
    top_p: float = 1
    top_k: int = 50
    min_new_tokens: int = 1
    max_new_tokens: int = 100
    api_endpoint: str = "https://workbench-api.res.ibm.com/v1"
    repetition_penalty: Optional[float] = None
    random_seed: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    truncate_input_tokens: Optional[int] = None

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompts: List[str] | str,
        stop: Optional[List[str]] = None,
        # stop: Optional[List[str]] = None,
        # run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> List[str] | str:
        if isinstance(prompts, str):
            prompts = [prompts]
        creds = genai.credentials.Credentials(api_key=self.api_key)
        gen_params = genai.schemas.generate_params.GenerateParams(
            decoding_method=self.decoding_method,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_new_tokens=self.min_new_tokens,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
            random_seed=self.random_seed,
            stop_sequences=self.stop_sequences,
            truncate_input_tokens=self.truncate_input_tokens,
        )
        model = genai.model.Model(
            model=self.model_name, params=gen_params, credentials=creds
        )
        out = model.generate(prompts=prompts)
        if len(out) == 1:
            return out[0].generated_text
        return [elem.generated_text for elem in out]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "decoding_method": self.decoding_method,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_new_tokens": self.min_new_tokens,
            "max_new_tokens": self.max_new_tokens,
            "api_key": self.api_key,
            "repetition_penalty": self.repetition_penalty,
            "random_seed": self.random_seed,
            "stop_sequences": self.stop_sequences,
            "truncate_input_tokens": self.truncate_input_tokens,
        }
