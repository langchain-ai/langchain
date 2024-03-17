import runhouse as rh
from langchain.prompts import PromptTemplate
from langchain_community.llms.self_hosted_hugging_face import LangchainLLMModelPipeline
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Optional, Any, List


def call_module(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
) -> str:
    if not self.pipeline_ref:
        self._pipeline = self.model_load_fn()
    return self.inference_fn(
        pipeline=self._pipeline, prompt=prompt, stop=stop, **kwargs
    )


if __name__ == '__main__':
    # For an on-demand A10G with the cheapest provider (default)
    gpu = rh.cluster(name='sasha-rh-a10x', instance_type='g5.2xlarge', provider='aws', region='eu-central-1')
    gpu.up_if_not()
    
    gpu.run(commands=["pip install runhouse"])
    gpu.run(commands=["pip uninstall -y runhouse"])
    gpu.run(commands=[
        "pip install git+https://github.com/run-house/runhouse.git@sb/fixes_langchain_integration#egg=runhouse"])


    model_env = rh.env(
        name="model_env15",
        reqs=["transformers", "torch"],
        secrets=["huggingface"]  # need to download  google/gemma-2b-it
    ).to(system=gpu, force_install=True)

    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate.from_template(template)

    # load_transformer_remote = rh.function(fn=_load_transformer).to(system=gpu, env=model_env)
    # generate_text_remote = rh.function(_generate_text).to(system=gpu, env=model_env)
    llm_module_pipeline_remote = LangchainLLMModelPipeline().to(system=gpu, env=model_env)

    print(gpu.status())

    # self_hosted_llm = SelfHostedHuggingFaceLLM(model_id="gemma-2b-it",
    #                                            llm_model_pipeline=llm_module_pipeline_remote)
    #
    # llm_chain = LLMChain(prompt=prompt, llm=self_hosted_llm)
    #
    # question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
    #
    # llm_chain.run(question)
