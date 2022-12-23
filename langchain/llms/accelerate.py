# TODO Garbage collect loaded models on exit?!?
# TODO Get Contextsizes for model IDs to impleme max_length=-1
# TODO Implement bf16 and model nativ types
# TODO Update gitignore or find better place to save model_weights

from typing import Any, List, Mapping, Optional
from pydantic import BaseModel, Extra
from langchain.llms.base import BaseLLM, LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import os

DEFAULT_MODEL_NAME = "EleutherAI/gpt-neox-20b" # nme as of Huggingface


class Accelerate(LLM, BaseModel):
    """Interference on big models on consumer hardware with the help off
    accelerate system. It distributes models that dont fit in video ram over multiple gpus
    and system ram, and if needed to disk (not yet implemented).
    A locl copy of the weights for much faster loading than hugging face standard is created.

    Tested with 
    "EleutherAI/gpt-neox-20b", "EleutherAI/gpt-j-6B", "facebook/opt-30b"

    30b model needs ~64gb = RAM+VRM and on an RTX 3080 16GB and 64 GB RAM it takes 10 Sek per token.

    Example:
        .. code-block:: python

            from langchain.llms import Accelerate
            model_name = "facebook/opt-30b"
            FastLLM = Accelerate.from_model_name(model_name=model_name)
            print(FastLLM("Hello World"))

    """

    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    tokenizer:Any = None
    """Tokenizer to use."""
    model:Any = None
    """model to use."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @staticmethod
    def get_accelerated_gpt(model_name:str = "EleutherAI/gpt-neox-20b"):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
        import torch

        #what else models are available and compatible? TODO

        weights_path = "./accelerated_model_weights/"+model_name
        if not os.path.exists(weights_path):
            model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16)
            os.makedirs(weights_path)
            # here an even faster method for saving and loading is described
            # https://www.philschmid.de/deploy-gptj-sagemaker
            model.save_pretrained(weights_path)
            

        config = AutoConfig.from_pretrained(model_name)

        config.use_cache = False

        with init_empty_weights():
            model_dummy = AutoModelForCausalLM.from_config(config)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model=model_dummy

        #bf16 would be possible too, but hardwaresupport would have to be tested before TODO

        #careful, if for example in a notebook and the model is already one time in the memmory, now it will
        #be loaded for second time and GPU is full, or even CPU and it may offload to ram or disk. Reatarting kernel helps.
        device_map = infer_auto_device_map(model_dummy, no_split_module_classes=["GPTNeoXLayer"],dtype=torch.float16)
        load_checkpoint_and_dispatch(
            model,
            weights_path,
            device_map=device_map,
            dtype="float16",
            offload_folder=None,
            offload_state_dict=False
        )
        return model

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        **kwargs: Any,
    ) -> LLM:
        """Construct the pipeline object from model_id and task."""
        try:
            from transformers import (
                AutoTokenizer,
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = cls.get_accelerated_gpt(model_name)
            return cls(
                tokenizer=tokenizer,
                model=model,
                **kwargs,
            )
        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please it install it with `pip install transformers`."
            )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_name": self.model_name},
        }

    @property
    def _llm_type(self) -> str:
        return "accelerated_LLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        input_tokenized = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(input_tokenized["input_ids"].to(0), do_sample=True,max_length=100,temperature=0.9,top_k=50,top_p=0.9)
        text = self.tokenizer.decode(output[0].tolist())
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, stop)
        return text