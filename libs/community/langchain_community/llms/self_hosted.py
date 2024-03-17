import importlib.util
import logging
import pickle
from typing import Any, Callable, List, Mapping, Optional
import runhouse as rh

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)


def _generic_interface_fn(
        self,
        pipeline: Any,
        prompt: str,
        *args: Any,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
) -> str:
    """Inference function to send to the remote hardware.

    Accepts a pipeline callable (or, more likely,
    a key pointing to the model on the cluster's object store)
    and returns text predictions for each document
    in the batch.
    """
    text = pipeline(prompt, *args, **kwargs)
    if stop is not None:
        text = enforce_stop_tokens(text, stop)
    return text


class ModelPipeline(rh.Module):

    def __init__(self, load_module_fn: Callable, interface_fn: Optional[Callable] = _generic_interface_fn):
        super().__init__()
        self._load_module_fn = load_module_fn
        self._interface_fn = interface_fn


class SelfHostedPipeline(LLM):
    """Model inference on self-hosted remote hardware.

    Supported hardware includes auto-launched instances on AWS, GCP, Azure,
    and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another
    cloud like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Example for custom pipeline and inference functions:
        .. code-block:: python

            from langchain_community.llms import SelfHostedPipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import runhouse as rh

            gpu = rh.ondemand_cluster(name="rh-a10x", instance_type="g5.2xlarge")
            my_env = rh.env(reqs=["transformers", "torch"])

            def load_pipeline():
                tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
                model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
                return pipeline(
                    "text-generation", model=model, tokenizer=tokenizer,
                    max_new_tokens=10
                )
            load_pipeline_remote = rh.function(fn=load_pipeline).to(gpu, env=model_env)

            def inference_fn(pipeline, prompt, stop = None):
                return pipeline(prompt)[0]["generated_text"]
            inference_fn_remote = rh.function(fn=inference_fn).to(gpu, env=model_env)

            llm = SelfHostedHuggingFaceLLM(
                model_load_fn=load_pipeline_remote, inference_fn=inference_fn_remote).to(gpu, env=model_env)

    """
    pipeline_ref: Any  #: :meta private:
    client: Any  #: :meta private:
    # inference_fn: rh.Function  #: :meta private:
    # """Inference function to send to the remote hardware."""
    # model_load_fn: rh.Function
    # """Function to load the model remotely on the server."""
    load_fn_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model load function."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow
        allow_population_by_field_name = True
        arbitrary_types_allowed = True

    def __init__(self, model_pipeline: ModelPipeline, **kwargs: Any):
        """Init the pipeline with an auxiliary function.

        The load function must be in global scope to be imported
        and run on the server, i.e. in a module and not a REPL or closure.
        Then, initialize the remote inference function.
        """
        super().__init__(**kwargs)
        _load_fn_kwargs = self.load_fn_kwargs or {}
        self.pipeline_ref = model_pipeline.pipeline_ref
        self.client = model_pipeline.inference_fn
        self.hardware = model_pipeline.system
        self.env = model_pipeline.env

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"hardware": self.hardware},
        }

    @property
    def _llm_type(self) -> str:
        return "self_hosted_llm"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        if not self.pipeline_ref:
            self._pipeline = self.model_load_fn.remote()
        return self.inference_fn(
            pipeline=self._pipeline, prompt=prompt, stop=stop, **kwargs
        )
