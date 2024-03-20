import runhouse as rh
from langchain_community.llms import SelfHostedHuggingFaceLLM

if __name__ == '__main__':
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    # For an on-demand A10G with the cheapest provider (default)
    gpu = rh.cluster(name='sasha-rh-a10x', instance_type='g5.8xlarge', provider='aws', region='eu-central-1')
    model_env = rh.env(
        name="model_env15",
        reqs=["transformers", "torch", "accelerate", "huggingface-hub", "langchain"],
        secrets=["huggingface"]  # need to download  google/gemma-2b-it
    )
    gpu.run(commands=["pip install langchain"])

    self_hosted_llm = SelfHostedHuggingFaceLLM(model_id="google/gemma-2b-it", task="text2text-generation", hardware=gpu, env=model_env)

    # template = """Question: {question}
    #
    # Answer: Let's think step by step."""

    # template = """Question: {question}
    #
    # Answer: Let's think step by step."""
    # prompt = PromptTemplate.from_template(template)
    #
    # llm_chain = LLMChain(prompt=prompt, llm=self_hosted_llm)
    #
    #
    #
    # print(llm_chain.run("What is the capital of Germany?"))

    question = "What is the capital of Germany?"

    print(self_hosted_llm(question))
