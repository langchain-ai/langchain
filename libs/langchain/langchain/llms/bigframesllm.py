from langchain.llms.base import BaseLLM, CallbackManagerForLLMRun
from typing import Any, Dict, List, Optional
import bigframes
from langchain.pydantic_v1 import root_validator
from langchain.schema import (
    Generation,
    LLMResult,
)

_TEXT_GENERATE_RESULT_COLUMN = "ml_generate_text_llm_result"

class BigFramesLLM(BaseLLM):
    """BigFrames large language models.
    """
     
    session: Optional[bigframes.Session] = None,
    connection: Optional[str] = None,
    model_name = "PaLM2TextGenerator"
    "Underlying model name."
    temperature: float = 0.0
    "Sampling temperature, it controls the degree of randomness in token selection."
    max_output_tokens: int = 128
    "Token limit determines the maximum amount of text output from one prompt."
    top_p: float = 0.95
    "Tokens are selected from most probable to least until the sum of their "
    "probabilities equals the top-p value. Top-p is ignored for Codey models."
    top_k: int = 40
    "How the model selects tokens for output, the next token is selected from "
    "among the top-k most probable tokens. Top-k is ignored for Codey models."

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            from bigframes.ml.llm import PaLM2TextGenerator
        except ImportError:
            raise ImportError(
                "Could not import bigframes.ml.llm python package. "
                "Please install it with `pip install bigframes`."
            )
        
        values["client"] = PaLM2TextGenerator(
            session=values["session"],
            connection_name=values["connection"],
        )
        return values
    
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling bigframesllm."""
        return {
            "session": self.session,
            "connection": self.connection,
            "model_name": self.model_name,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompts_df = bigframes.pandas.DataFrame({"index": [prompt]})
        response = self.client.predict(X=prompts_df,
                                        temperature=self.temperature, 
                                        max_output_tokens= self.max_output_tokens,
                                        top_k = self.top_k,
                                        top_p = self.top_p)
        text = response[_TEXT_GENERATE_RESULT_COLUMN].to_pandas()[0]
        return text
        

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        prompts_df = bigframes.pandas.DataFrame({"index": prompts})
        responses_df = self.client.predict(X=prompts_df,
                                        temperature=self.temperature, 
                                        max_output_tokens= self.max_output_tokens,
                                        top_k = self.top_k,
                                        top_p = self.top_p)
        generations: List[List[Generation]] = []
        results_pd = responses_df[_TEXT_GENERATE_RESULT_COLUMN].to_pandas()
        for result in results_pd:
            generations.append([Generation(text=result)])
        return LLMResult(generations=generations)
    
    @property
    def _llm_type(self) -> str:
        return "bigframesllm"
        

    
