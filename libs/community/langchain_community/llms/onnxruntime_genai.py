from typing import Any, Dict, Optional
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.utils import pre_init
from pydantic import Field

class OnnxruntimeGenAi(BaseLLM):
    """Onnxruntime GenAI model.

    To use, you should have the onnxruntime-genai (CPU/DML/CUDA) library installed, and provide the
    path to the ONNX model as a named parameter to the constructor.
    Check out: https://github.com/microsoft/onnxruntime-genai

    Example:
        .. code-block:: python

            from langchain_community.llms import OnnxruntimeGenAi
            llm = OnnxruntimeGenAi(model_path="/path/to/onnx/model")
    """

    model_path: str
    """The path to the Onnx model file."""
    
    max_tokens: Optional[int] = 256
    """The maximum number of tokens to generate."""

    temperature: Optional[float] = 0.8
    """The temperature to use for sampling."""

    top_p: Optional[float] = 0.95
    """The top-p value to use for sampling."""

    top_k: Optional[int] = 40
    """The top-k value to use for sampling."""

    do_sample: Optional[bool] = False
    """The do_sample value to do sampling or not."""

    n_batch: Optional[int] = Field(8, alias="n_batch")
    """Number of tokens to process in parallel.
    Should be a number between 1 and n_ctx."""

    repeat_penalty: Optional[float] = 1.1
    """The penalty to apply to repeated tokens."""

    verbose: bool = True
    """Print verbose output to stderr."""

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            from onnxruntime_genai import GeneratorParams,Model,Tokenizer,Generator
        except ImportError:
            raise ImportError(
                "Could not import onnxruntime-genai python package. "
                "Please install it with `pip install onnxruntime-genai`."
            )
 
        try:
            model_path = values["model_path"]
            values["client"] = Model(model=values["model"])
        except Exception as e:
            raise ValueError(
                f"Could not load Onnx model from path: {model_path}. "
                f"Received error {e}"
            )
        
        try:
            model = values["client"]
            values["tokenizer"] = Tokenizer(model)
            values["tokenizer_stream"] = values["tokenizer"].create_stream()
        except Exception as e:
            raise ValueError(
                f"Could not load Onnx model from path: {model_path}. "
                f"Received error {e}"
            )

        return values
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling onnxruntime-genai."""
        return {
            "do_sample": self.do_sample,
            "max_length": self.max_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty":self.repeat_penalty,
            "batch_size":self.n_batch
        }
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "onnruntime-genai"
    
    def _generate(self, prompts, stop = None, run_manager = None, **kwargs):

        from onnxruntime_genai import GeneratorParams,Generator
        text_generations: list[str] = []
        answer:str=""

        # Encode prompts
        input_token = self.tokenizer.encode_batch(prompts)
        search_options = self._default_params

        # Build generator params
        params = GeneratorParams(self.model)
        params.set_search_options(**search_options)
        generator = Generator(self.model, params)

        # Append input token
        generator.append_tokens(input_token)
        while not generator.is_done():
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]

            # TODO: Verify EOS token in here
            answer+=self.tokenizer.decode(new_token)

            print(self.tokenizer_stream.decode(new_token), end='', flush=True)
        text_generations.append(answer)

        del generator
        
        return LLMResult(generations=[[Generation(text=text) for text in text_generations]])
