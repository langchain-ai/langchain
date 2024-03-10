import importlib.util
import logging
import pickle
import runhouse as rh
from typing import Any, List, Mapping, Optional
from pydantic import Field


from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "google/gemma-2b-it"

def _generate_text(
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


def _send_pipeline_to_device(pipeline: Any, device: int) -> Any:
    """Send a pipeline to a device on the cluster."""
    if isinstance(pipeline, str):
        with open(pipeline, "rb") as f:
            pipeline = pickle.load(f)

    if importlib.util.find_spec("torch") is not None:
        import torch

        cuda_device_count = torch.cuda.device_count()
        if device < -1 or (device >= cuda_device_count):
            raise ValueError(
                f"Got device=={device}, "
                f"device is required to be within [-1, {cuda_device_count})"
            )
        if device < 0 and cuda_device_count > 0:
            logger.warning(
                "Device has %d GPUs available. "
                "Provide device={deviceId} to `from_model_id` to use available"
                "GPUs for execution. deviceId is -1 for CPU and "
                "can be a positive integer associated with CUDA device id.",
                cuda_device_count,
            )

        pipeline.device = torch.device(device)
        pipeline.model = pipeline.model.to(pipeline.device)
    return pipeline


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

    model_id: str = DEFAULT_MODEL_ID
    device: int = 0
    """Device to use for inference. -1 for CPU, 0 for GPU, 1 for second GPU, etc."""
    inference_fn: rh.Function  #: :meta private:
    """Inference function to send to the remote hardware."""
    model_load_fn: rh.Function
    """Function to load the model remotely on the server."""
    load_fn_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model load function."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def __init__(self, **kwargs: Any):
        """Init the pipeline with an auxiliary function.

        The load function must be in global scope to be imported
        and run on the server, i.e. in a module and not a REPL or closure.
        Then, initialize the remote inference function.
        """
        super().__init__(**kwargs)

        # remote_load_fn = self.model_load_fn
        # _load_fn_kwargs = self.load_fn_kwargs or {}
        # # self.pipeline_ref = remote_load_fn.remote(**_load_fn_kwargs)

        #self.inference_fn = self.inference_fn

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"hardware": self.system},
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
