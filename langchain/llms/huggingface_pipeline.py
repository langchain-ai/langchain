"""Wrapper around HuggingFace Pipeline APIs."""
import importlib.util
import logging
from typing import Any, List, Mapping, Optional

from pydantic import BaseModel, Extra

from langchain.llms.base import LLM

DEFAULT_MODEL_ID = "gpt2"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text2text-generation", "text-generation")

logger = logging.getLogger()

class HuggingFaceTokenProcessor:
    '''A LogitsProcessor to handle stop sequences and streaming.'''

    def __init__(self, llm, tokenizer, stop, eos_token_id, generation_offset):
        self.llm = llm
        self.tokenizer = tokenizer
        self.stop = stop
        self.eos_token_id = eos_token_id
        self.generation_offset = generation_offset
        self.pre_lens = None
        self.cur_seqs = None
        self.stop_lens = None

    def generate(self, *params, **kwparams):
        # call pipeline which will then pass token ids to _logits_processor
        outputs = self.llm.pipeline(
            *params,
            logits_processor = [self._logits_processor],
            **kwparams
        )
        # trim output
        batchsize = len(outputs)
        seqs = [
            outputs[batch_idx]["generated_text"][self.generation_offset :][: self.stop_lens[batch_idx]]
            for batch_idx in range(batchsize)
        ]
        # process any terminating token, which isn't passed to a logits_processor
        self._update_seqs(seqs)
        # stream the last delayed token
        self._finish_seqs()
        return seqs

    def _logits_processor(self, input_ids, logits):
        '''Decodes the latest input_ids and hands them off to _update_seqs to process.'''
        # for now this unneccessarily re-decodes the same starting logits repeatedly
        seqs = [
            seq[self.generation_offset:]
            for seq in self.tokenizer.batch_decode(input_ids)
        ]
        return self._update_seqs(seqs, logits)

    def _update_seqs(self, new_seqs, logits=None):
        '''Performs streaming and processes stop tokens.
           Tokens are streamed with a 1-token delay to ensure the decoder has trailing context.
           Sequences are ended by causing generation of eos_token_id.'''
        batchsize = len(new_seqs)
        if self.cur_seqs is None:
            # init data now we know the batchsize
            # data is parallel lists for now
            self.pre_lens = [0] * batchsize
            self.cur_seqs = [''] * batchsize
            self.stop_lens = [None] * batchsize
        # update information on delayed tokens
        pre_pre_lens = self.pre_lens
        self.pre_lens = [len(pre_seq) for pre_seq in self.cur_seqs]
        self.cur_seqs = new_seqs

        for batch_idx in range(len(self.cur_seqs)):
            # data for each batch index
            pre_pre_len = pre_pre_lens[batch_idx]
            pre_len = self.pre_lens[batch_idx]
            seq = self.cur_seqs[batch_idx]
            stop_len = self.stop_lens[batch_idx]

            # stream
            if self.llm.streaming and pre_pre_lens:
                token = seq[: stop_len][pre_pre_len : pre_len]
                if len(token):
                    self.llm.callback_manager.on_llm_new_token(
                        token,
                        verbose=self.llm.verbose,
                        batch_idx=batch_idx
                    )

            # stop
            if self.stop is not None and logits is not None:
                for stop in self.stop:
                    start_offset = max(0, pre_pre_len - (len(stop) - 1))
                    found_offset = seq.find(stop, start_offset)
                    if found_offset >= 0:
                        logits[batch_idx][self.eos_token_id] = float('inf')
                        self.stop_lens[batch_idx] = found_offset

        return logits

    def _finish_seqs(self):
        '''Streams out the last delayed token.'''
        if self.llm.streaming and self.pre_lens:
            for batch_idx in range(len(self.cur_seqs)):
                seq = self.cur_seqs[batch_idx]
                pre_len = self.pre_lens[batch_idx]
                stop_len = self.stop_lens[batch_idx]
                token = seq[pre_len : stop_len]
                if len(token):
                    self.llm.callback_manager.on_llm_new_token(
                        token,
                        verbose=self.llm.verbose,
                        batch_idx=batch_idx
                    )


class HuggingFacePipeline(LLM, BaseModel):
    """Wrapper around HuggingFace Pipeline API.

    To use, you should have the ``transformers`` python package installed.

    Only supports `text-generation` and `text2text-generation` for now.

    Example using from_model_id:
        .. code-block:: python

            from langchain.llms.huggingface_pipeline import HuggingFacePipeline
            hf = HuggingFacePipeline.from_model_id(
                model_id="gpt2", task="text-generation"
            )
    Example passing pipeline in directly:
        .. code-block:: python

            from langchain.llms.huggingface_pipeline import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_id = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
            )
            hf = HuggingFacePipeline(pipeline=pipe)
    """

    pipeline: Any  #: :meta private:
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""
    streaming: bool = False
    """Whether to stream the results or not."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        task: str,
        device: int = -1,
        model_kwargs: Optional[dict] = None,
        pipeline_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLM:
        """Construct the pipeline object from model_id and task."""
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
            from transformers import pipeline as hf_pipeline

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please it install it with `pip install transformers`."
            )

        _model_kwargs = model_kwargs or {}
        _pipeline_kwargs = pipeline_kwargs or {}
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
                    "GPUs for execution. deviceId is -1 (default) for CPU and "
                    "can be a positive integer associated with CUDA device id.",
                    cuda_device_count,
                )

        pipeline = hf_pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_kwargs=_model_kwargs,
            **_pipeline_kwargs,
        )
        if pipeline.task not in VALID_TASKS:
            raise ValueError(
                f"Got invalid task {pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        return cls(
            pipeline=pipeline,
            model_id=model_id,
            model_kwargs=_model_kwargs,
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
        return "huggingface_pipeline"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if self.pipeline.task == "text-generation":
            # Text generation return includes the starter text.
            generation_offset = len(prompt)
        elif self.pipeline.task == "text2text-generation":
            generation_offset = 0
        else:
            raise ValueError(
                f"Got invalid task {self.pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )

        token_processor = HuggingFaceTokenProcessor(
            llm=self,
            tokenizer=self.pipeline.tokenizer,
            stop=stop,
            eos_token_id=self.pipeline.model.config.eos_token_id,
            generation_offset=generation_offset,
        )

        return token_processor.generate(prompt)[0]
