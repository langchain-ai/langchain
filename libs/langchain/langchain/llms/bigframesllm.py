from typing import Any, Dict, List, Optional, Union, cast, Tuple

import bigframes

import bigframes.pandas as bf
import langchain
import inspect

from langchain.llms.base import BaseLLM, CallbackManagerForLLMRun
from langchain.callbacks.base import BaseCallbackManager
from langchain.pydantic_v1 import root_validator
from langchain.load.dump import dumpd
from langchain.schema import (
    Generation,
    LLMResult,
    PromptValue,
    RunInfo,
)
from langchain.callbacks.manager import (
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)

_TEXT_GENERATE_RESULT_COLUMN = "ml_generate_text_llm_result"

def get_prompts(
    params: Dict[str, Any], prompts: List[str]
) -> Tuple[Dict[int, List], str, List[int], List[str]]:
    """Get prompts that are already cached."""
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    missing_prompts = []
    missing_prompt_idxs = []
    existing_prompts = {}
    for i, prompt in enumerate(prompts):
        if langchain.llm_cache is not None:
            cache_val = langchain.llm_cache.lookup(prompt, llm_string)
            if isinstance(cache_val, list):
                existing_prompts[i] = cache_val
            else:
                missing_prompts.append(prompt)
                missing_prompt_idxs.append(i)
    return existing_prompts, llm_string, missing_prompt_idxs, missing_prompts


# def update_cache(
#     existing_prompts: Dict[int, List],
#     llm_string: str,
#     missing_prompt_idxs: List[int],
#     new_results: LLMResult,
#     prompts: List[str],
# ) -> Optional[dict]:
#     """Update the cache and get the LLM output."""
#     for i, result in enumerate(new_results.generations):
#         existing_prompts[missing_prompt_idxs[i]] = result
#         prompt = prompts[missing_prompt_idxs[i]]
#         if langchain.llm_cache is not None:
#             langchain.llm_cache.update(prompt, llm_string, result)
#     llm_output = new_results.llm_output
#     return llm_output


class BigFramesLLM(BaseLLM):
    """BigFrames large language models."""

    session: Optional[bigframes.Session] = (None,)
    connection: Optional[str] = (None,)
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

    def __call__(
        self,
        prompt: Union[str, bf.Series, bf.DataFrame],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Union[bf.DataFrame, bf.Series]:
        if isinstance(prompt, str):
            prompts_df = bigframes.pandas.Series([prompt])
            response = self.client.predict(
                X=prompts_df,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            # text = response[_TEXT_GENERATE_RESULT_COLUMN].to_pandas()[0]
            return response
        else:
            response = self.client.predict(
                X=prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            return response

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        prompts_df = bigframes.pandas.DataFrame({"index": prompts})
        responses_df = self.client.predict(
            X=prompts_df,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        generations: List[List[Generation]] = []
        results_pd = responses_df[_TEXT_GENERATE_RESULT_COLUMN].to_pandas().sort_index()
        for result in results_pd:
            generations.append([Generation(text=result)])
        return LLMResult(generations=generations)
    

    # def generate_prompt(
    #     self,
    #     prompts: List[PromptValue],
    #     stop: Optional[List[str]] = None,
    #     callbacks: Optional[Union[Callbacks, List[Callbacks]]] = None,
    #     **kwargs: Any,
    # ) -> bf.DataFrame:
    #     prompt_strings = [p.to_string() for p in prompts]
    #     return self.generate(prompt_strings, stop=stop, callbacks=callbacks, **kwargs)
    

    # def _generate_helper(
    #     self,
    #     prompts: List[str],
    #     stop: Optional[List[str]],
    #     run_managers: List[CallbackManagerForLLMRun],
    #     new_arg_supported: bool,
    #     **kwargs: Any,
    # ) -> bf.DataFrame:
    #     try:
    #         output = (
    #             self._generate(
    #                 prompts,
    #                 stop=stop,
    #                 # TODO: support multiple run managers
    #                 run_manager=run_managers[0] if run_managers else None,
    #                 **kwargs,
    #             )
    #             if new_arg_supported
    #             else self._generate(prompts, stop=stop)
    #         )
    #     except BaseException as e:
    #         for run_manager in run_managers:
    #             run_manager.on_llm_error(e)
    #         raise e
    #     # flattened_outputs = output.flatten()
    #     # for manager, flattened_output in zip(run_managers, flattened_outputs):
    #     #     manager.on_llm_end(flattened_output)
    #     if run_managers:
    #         output.run = [
    #             RunInfo(run_id=run_manager.run_id) for run_manager in run_managers
    #         ]
    #     return output

    # def generate(
    #     self,
    #     prompts: List[str],
    #     stop: Optional[List[str]] = None,
    #     callbacks: Optional[Union[Callbacks, List[Callbacks]]] = None,
    #     *,
    #     tags: Optional[Union[List[str], List[List[str]]]] = None,
    #     metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    #     run_name: Optional[Union[str, List[str]]] = None,
    #     **kwargs: Any,
    # ) -> bf.DataFrame:
    #     """Run the LLM on the given prompt and input."""
    #     if not isinstance(prompts, list):
    #         raise ValueError(
    #             "Argument 'prompts' is expected to be of type List[str], received"
    #             f" argument of type {type(prompts)}."
    #         )
    #     # Create callback managers
    #     if (
    #         isinstance(callbacks, list)
    #         and callbacks
    #         and (
    #             isinstance(callbacks[0], (list, BaseCallbackManager))
    #             or callbacks[0] is None
    #         )
    #     ):
    #         # We've received a list of callbacks args to apply to each input
    #         assert len(callbacks) == len(prompts)
    #         assert tags is None or (
    #             isinstance(tags, list) and len(tags) == len(prompts)
    #         )
    #         assert metadata is None or (
    #             isinstance(metadata, list) and len(metadata) == len(prompts)
    #         )
    #         assert run_name is None or (
    #             isinstance(run_name, list) and len(run_name) == len(prompts)
    #         )
    #         callbacks = cast(List[Callbacks], callbacks)
    #         tags_list = cast(List[Optional[List[str]]], tags or ([None] * len(prompts)))
    #         metadata_list = cast(
    #             List[Optional[Dict[str, Any]]], metadata or ([{}] * len(prompts))
    #         )
    #         run_name_list = run_name or cast(
    #             List[Optional[str]], ([None] * len(prompts))
    #         )
    #         callback_managers = [
    #             CallbackManager.configure(
    #                 callback,
    #                 self.callbacks,
    #                 self.verbose,
    #                 tag,
    #                 self.tags,
    #                 meta,
    #                 self.metadata,
    #             )
    #             for callback, tag, meta in zip(callbacks, tags_list, metadata_list)
    #         ]
    #     else:
    #         # We've received a single callbacks arg to apply to all inputs
    #         callback_managers = [
    #             CallbackManager.configure(
    #                 cast(Callbacks, callbacks),
    #                 self.callbacks,
    #                 self.verbose,
    #                 cast(List[str], tags),
    #                 self.tags,
    #                 cast(Dict[str, Any], metadata),
    #                 self.metadata,
    #             )
    #         ] * len(prompts)
    #         run_name_list = [cast(Optional[str], run_name)] * len(prompts)

    #     params = self.dict()
    #     params["stop"] = stop
    #     options = {"stop": stop}
    #     (
    #         existing_prompts,
    #         llm_string,
    #         missing_prompt_idxs,
    #         missing_prompts,
    #     ) = get_prompts(params, prompts)
    #     disregard_cache = self.cache is not None and not self.cache
    #     new_arg_supported = inspect.signature(self._generate).parameters.get(
    #         "run_manager"
    #     )
    #     new_arg_supported = False

    #     if langchain.llm_cache is None or disregard_cache:
    #         if self.cache is not None and self.cache:
    #             raise ValueError(
    #                 "Asked to cache, but no cache found at `langchain.cache`."
    #             )
    #         run_managers = [
    #             callback_manager.on_llm_start(
    #                 dumpd(self),
    #                 [prompt],
    #                 invocation_params=params,
    #                 options=options,
    #                 name=run_name,
    #             )[0]
    #             for callback_manager, prompt, run_name in zip(
    #                 callback_managers, prompts, run_name_list
    #             )
    #         ]
    #         output = self._generate_helper(
    #             prompts, stop, run_managers, bool(new_arg_supported), **kwargs
    #         )
    #         return output
    #     if len(missing_prompts) > 0:
    #         run_managers = [
    #             callback_managers[idx].on_llm_start(
    #                 dumpd(self),
    #                 [prompts[idx]],
    #                 invocation_params=params,
    #                 options=options,
    #                 name=run_name_list[idx],
    #             )[0]
    #             for idx in missing_prompt_idxs
    #         ]
    #         results = self._generate_helper(
    #             missing_prompts, stop, run_managers, bool(new_arg_supported), **kwargs
    #         )
    #         # llm_output = update_cache(
    #         #     existing_prompts, llm_string, missing_prompt_idxs, new_results, prompts
    #         # )
    #         # run_info = (
    #         #     [RunInfo(run_id=run_manager.run_id) for run_manager in run_managers]
    #         #     if run_managers
    #         #     else None
    #         # )
    #     else:
    #         llm_output = {}
    #         # run_info = None
    #         results = bf.DataFrame()
    #     return results
    #     # generations = [existing_prompts[i] for i in range(len(prompts))]
    #     # return LLMResult(generations=generations, llm_output=llm_output, run=run_info)

    @property
    def _llm_type(self) -> str:
        return "bigframesllm"
