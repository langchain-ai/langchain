"""Wrapper around HuggingFace Pipeline APIs."""
import importlib.util
import logging
from typing import Any, Callable, List, Mapping, Optional

from pydantic import BaseModel, Extra

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

DEFAULT_MODEL_ID = "EleutherAI/gpt-j-6B"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text2text-generation", "text-generation")

logger = logging.getLogger()


def _generate_text(pipeline: Any, prompt: str, stop: Optional[List[str]] = None) -> str:
    """Inference function to send to the remote hardware. Accepts a sentence_transformer model_id and
    returns a list of embeddings for each document in the batch.
    """
    response = pipeline(prompt)
    if pipeline.task == "text-generation":
        # Text generation return includes the starter text.
        text = response[0]["generated_text"][len(prompt) :]
    elif pipeline.task == "text2text-generation":
        text = response[0]["generated_text"]
    else:
        raise ValueError(
            f"Got invalid task {pipeline.task}, "
            f"currently only {VALID_TASKS} are supported"
        )
    if stop is not None:
        # This is a bit hacky, but I can't figure out a better way to enforce
        # stop tokens when making calls to huggingface_hub.
        text = enforce_stop_tokens(text, stop)
    return text


def _load_transformer(
    model_id: str = DEFAULT_MODEL_ID,
    task: str = DEFAULT_TASK,
    device: int = 0,
    model_kwargs: Optional[dict] = None,
) -> Any:
    """Inference function to send to the remote hardware. Accepts a huggingface model_id and
    returns a pipeline for the task.
    """
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers import pipeline as hf_pipeline

    _model_kwargs = model_kwargs or {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)

    try:
        if task == "text-generation":
            model = AutoModelForCausalLM.from_pretrained(model_id, **_model_kwargs)
        elif task == "text2text-generation":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **_model_kwargs)
        else:
            raise ValueError(
                f"Got invalid task {task}, "
                f"currently only {VALID_TASKS} are supported"
            )
    except ImportError as e:
        raise ValueError(
            f"Could not load the {task} model due to missing dependencies."
        ) from e

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

    pipeline = hf_pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_kwargs=_model_kwargs,
    )
    if pipeline.task not in VALID_TASKS:
        raise ValueError(
            f"Got invalid task {pipeline.task}, "
            f"currently only {VALID_TASKS} are supported"
        )
    return pipeline


class SelfHostedHuggingFacePipeline(LLM, BaseModel):
    """Wrapper around HuggingFace Pipeline API to perform inference on self-hosted remote hardware.
    Supported hardware includes auto-launched instances on AWS, GCP, Azure, and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another cloud like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Only supports `text-generation` and `text2text-generation` for now.

    Example using from_model_id:
        .. code-block:: python

            from langchain.llms import SelfHostedHuggingFacePipeline
            import runhouse as rh
            gpu = rh.cluster(name="rh-a10x", instance_type="A100:1")
            hf = SelfHostedHuggingFacePipeline.from_model_id(
                model_id="google/flan-t5-large", task="text2text-generation", hardware=gpu
            )
    Example passing fn that generates a pipeline (because the pipeline is not serializable):
        .. code-block:: python

            from langchain.llms import SelfHostedHuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import runhouse as rh

            def get_pipeline():
                model_id = "gpt2"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(model_id)
                pipe = pipeline(
                    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
                )
            hf = SelfHostedHuggingFacePipeline(model_load_fn=get_pipeline, model_id="gpt2", hardware=gpu)
    """

    pipeline_ref: Any  #: :meta private:
    client: Any  #: :meta private:
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""
    model_load_fn: Callable = _load_transformer
    """Function to load the model remotely on the server."""
    load_fn_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model load function."""
    model_reqs: List[str] = ["transformers", "torch"]
    """Requirements to install on hardware to inference the model."""
    hardware: Any
    """Remote hardware to send the inference function to."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def __init__(self, **kwargs: Any):
        """Construct the pipeline remotely using an auxiliary function sent to the server
        which loads and returns the pipeline. The load function needs to be importable to be imported
        and run on the server, i.e. in a module and not a REPL or closure.
        Then, initialize the remote inference function."""
        super().__init__(**kwargs)
        try:
            import runhouse as rh

        except ImportError:
            raise ValueError(
                "Could not import runhouse python package. "
                "Please install it with `pip install runhouse`."
            )

        remote_load_fn = rh.send(fn=self.model_load_fn).to(
            self.hardware, reqs=["pip:./"] + self.model_reqs
        )
        _load_fn_kwargs = self.load_fn_kwargs or {}
        self.pipeline_ref = remote_load_fn.remote(**_load_fn_kwargs)

        self.client = rh.send(fn=_generate_text).to(
            self.hardware, reqs=["pip:./"] + self.model_reqs
        )

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        task: str,
        hardware: Any,
        model_reqs: Optional[List[str]] = None,
        device: int = 0,
        model_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLM:
        """Construct the pipeline object from model_id and task."""

        load_fn_kwargs = {"model_id": model_id, "task": task, "device": device}
        return cls(
            model_id=model_id,
            model_kwargs=model_kwargs,
            load_fn_kwargs=load_fn_kwargs,
            hardware=hardware,
            model_reqs=["transformers", "torch"] + (model_reqs or []),
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_id": self.model_id},
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        return "selfhosted_huggingface_pipeline"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.client(pipeline=self.pipeline_ref, prompt=prompt, stop=stop)
