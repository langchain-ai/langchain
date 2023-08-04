import importlib.util
import logging
from typing import Any, Callable, List, Mapping, Optional

from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.self_hosted import SelfHostedPipeline
from langchain.llms.utils import enforce_stop_tokens

DEFAULT_MODEL_ID = "gpt2"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text2text-generation", "text-generation", "summarization")

logger = logging.getLogger(__name__)


def _generate_text(
    pipeline: Any,
    prompt: str,
    *args: Any,
    stop: Optional[List[str]] = None,
    **kwargs: Any,
) -> str:
    """Inference function to send to the remote hardware.

    Accepts a Hugging Face pipeline (or more likely,
    a key pointing to such a pipeline on the cluster's object store)
    and returns generated text.
    """
    response = pipeline(prompt, *args, **kwargs)
    if pipeline.task == "text-generation":
        # Text generation return includes the starter text.
        text = response[0]["generated_text"][len(prompt) :]
    elif pipeline.task == "text2text-generation":
        text = response[0]["generated_text"]
    elif pipeline.task == "summarization":
        text = response[0]["summary_text"]
    else:
        raise ValueError(
            f"Got invalid task {pipeline.task}, "
            f"currently only {VALID_TASKS} are supported"
        )
    if stop is not None:
        text = enforce_stop_tokens(text, stop)
    return text


def _load_transformer(
    model_id: str = DEFAULT_MODEL_ID,
    task: str = DEFAULT_TASK,
    device: int = 0,
    model_kwargs: Optional[dict] = None,
) -> Any:
    """Inference function to send to the remote hardware.

    Accepts a huggingface model_id and returns a pipeline for the task.
    """
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers import pipeline as hf_pipeline

    _model_kwargs = model_kwargs or {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)

    try:
        if task == "text-generation":
            model = AutoModelForCausalLM.from_pretrained(model_id, **_model_kwargs)
        elif task in ("text2text-generation", "summarization"):
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


class SelfHostedHuggingFaceLLM(SelfHostedPipeline):
    """HuggingFace Pipeline API to run on self-hosted remote hardware.

    Supported hardware includes auto-launched instances on AWS, GCP, Azure,
    and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another cloud
    like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Only supports `text-generation`, `text2text-generation` and `summarization` for now.

    Example using from_model_id:
        .. code-block:: python

            from langchain.llms import SelfHostedHuggingFaceLLM
            import runhouse as rh
            gpu = rh.cluster(name="rh-a10x", instance_type="A100:1")
            hf = SelfHostedHuggingFaceLLM(
                model_id="google/flan-t5-large", task="text2text-generation",
                hardware=gpu
            )
    Example passing fn that generates a pipeline (bc the pipeline is not serializable):
        .. code-block:: python

            from langchain.llms import SelfHostedHuggingFaceLLM
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import runhouse as rh

            def get_pipeline():
                model_id = "gpt2"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(model_id)
                pipe = pipeline(
                    "text-generation", model=model, tokenizer=tokenizer
                )
                return pipe
            hf = SelfHostedHuggingFaceLLM(
                model_load_fn=get_pipeline, model_id="gpt2", hardware=gpu)
    """

    model_id: str = DEFAULT_MODEL_ID
    """Hugging Face model_id to load the model."""
    task: str = DEFAULT_TASK
    """Hugging Face task ("text-generation", "text2text-generation" or
    "summarization")."""
    device: int = 0
    """Device to use for inference. -1 for CPU, 0 for GPU, 1 for second GPU, etc."""
    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""
    hardware: Any
    """Remote hardware to send the inference function to."""
    model_reqs: List[str] = ["./", "transformers", "torch"]
    """Requirements to install on hardware to inference the model."""
    model_load_fn: Callable = _load_transformer
    """Function to load the model remotely on the server."""
    inference_fn: Callable = _generate_text  #: :meta private:
    """Inference function to send to the remote hardware."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def __init__(self, **kwargs: Any):
        """Construct the pipeline remotely using an auxiliary function.

        The load function needs to be importable to be imported
        and run on the server, i.e. in a module and not a REPL or closure.
        Then, initialize the remote inference function.
        """
        load_fn_kwargs = {
            "model_id": kwargs.get("model_id", DEFAULT_MODEL_ID),
            "task": kwargs.get("task", DEFAULT_TASK),
            "device": kwargs.get("device", 0),
            "model_kwargs": kwargs.get("model_kwargs", None),
        }
        super().__init__(load_fn_kwargs=load_fn_kwargs, **kwargs)

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
