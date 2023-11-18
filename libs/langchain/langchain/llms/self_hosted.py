import importlib.util
import logging
import pickle
from typing import Any, Callable, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.pydantic_v1 import Extra

logger = logging.getLogger(__name__)


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

            from langchain.llms import SelfHostedPipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import runhouse as rh

            def load_pipeline():
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                model = AutoModelForCausalLM.from_pretrained("gpt2")
                return pipeline(
                    "text-generation", model=model, tokenizer=tokenizer,
                    max_new_tokens=10
                )
            def inference_fn(pipeline, prompt, stop = None):
                return pipeline(prompt)[0]["generated_text"]

            gpu = rh.cluster(name="rh-a10x", instance_type="A100:1")
            llm = SelfHostedPipeline(
                model_load_fn=load_pipeline,
                hardware=gpu,
                model_reqs=model_reqs, inference_fn=inference_fn
            )
    Example for <2GB model (can be serialized and sent directly to the server):
        .. code-block:: python

            from langchain.llms import SelfHostedPipeline
            import runhouse as rh
            gpu = rh.cluster(name="rh-a10x", instance_type="A100:1")
            my_model = ...
            llm = SelfHostedPipeline.from_pipeline(
                pipeline=my_model,
                hardware=gpu,
                model_reqs=["./", "torch", "transformers"],
            )
    Example passing model path for larger models:
        .. code-block:: python

            from langchain.llms import SelfHostedPipeline
            import runhouse as rh
            import pickle
            from transformers import pipeline

            generator = pipeline(model="gpt2")
            rh.blob(pickle.dumps(generator), path="models/pipeline.pkl"
                ).save().to(gpu, path="models")
            llm = SelfHostedPipeline.from_pipeline(
                pipeline="models/pipeline.pkl",
                hardware=gpu,
                model_reqs=["./", "torch", "transformers"],
            )
    """

    pipeline_ref: Any  #: :meta private:
    client: Any  #: :meta private:
    inference_fn: Callable = _generate_text  #: :meta private:
    """Inference function to send to the remote hardware."""
    hardware: Any
    """Remote hardware to send the inference function to."""
    model_load_fn: Callable
    """Function to load the model remotely on the server."""
    load_fn_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model load function."""
    model_reqs: List[str] = ["./", "torch"]
    """Requirements to install on hardware to inference the model."""

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
        try:
            import runhouse as rh

        except ImportError:
            raise ImportError(
                "Could not import runhouse python package. "
                "Please install it with `pip install runhouse`."
            )

        remote_load_fn = rh.function(fn=self.model_load_fn).to(
            self.hardware, reqs=self.model_reqs
        )
        _load_fn_kwargs = self.load_fn_kwargs or {}
        self.pipeline_ref = remote_load_fn.remote(**_load_fn_kwargs)

        self.client = rh.function(fn=self.inference_fn).to(
            self.hardware, reqs=self.model_reqs
        )

    @classmethod
    def from_pipeline(
        cls,
        pipeline: Any,
        hardware: Any,
        model_reqs: Optional[List[str]] = None,
        device: int = 0,
        **kwargs: Any,
    ) -> LLM:
        """Init the SelfHostedPipeline from a pipeline object or string."""
        if not isinstance(pipeline, str):
            logger.warning(
                "Serializing pipeline to send to remote hardware. "
                "Note, it can be quite slow"
                "to serialize and send large models with each execution. "
                "Consider sending the pipeline"
                "to the cluster and passing the path to the pipeline instead."
            )

        load_fn_kwargs = {"pipeline": pipeline, "device": device}
        return cls(
            load_fn_kwargs=load_fn_kwargs,
            model_load_fn=_send_pipeline_to_device,
            hardware=hardware,
            model_reqs=["transformers", "torch"] + (model_reqs or []),
            **kwargs,
        )

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
        return self.client(
            pipeline=self.pipeline_ref, prompt=prompt, stop=stop, **kwargs
        )
