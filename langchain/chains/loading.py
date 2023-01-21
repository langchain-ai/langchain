from langchain.chains.llm import LLMChain
from langchain.llms.loading import load_llm_from_config, load_llm
from langchain.prompts.loading import load_prompt, load_prompt_from_config

def _load_llm_chain(config: dict):
    """Load LLM chain from config dict."""
    llm_config = config.pop("llm")
    llm = load_llm_from_config(llm_config)
    prompt_config = config.pop("prompt")
    prompt = load_prompt_from_config(prompt_config)
    return LLMChain(llm=llm, prompt=prompt, **config)


type_to_loader_dict = {
    "llm_chain": _load_llm_chain
}

def load_chain_from_config(config: dict):
    """Load chain from Config Dict."""
    if "_type" not in config:
        raise ValueError("Must specify an chain Type in config")
    config_type = config.pop("_type")

    if config_type not in type_to_loader_dict:
        raise ValueError(f"Loading {config_type} chain not supported")

    chain_loader= type_to_loader_dict[config_type]
    return chain_loader(config)