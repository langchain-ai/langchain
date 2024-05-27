# Language Models

The Language Models module is responsible for managing the large language models (LLMs) used by the Controller.

Currently, the framework supports the following LLMs:
- GPT-4 / GPT-3.5 (Remote - OpenAI API)
- Llama-2 (Local - HuggingFace Transformers) 

The following sections describe how to instantiate individual LLMs and how to add new LLMs to the framework.

## LLM Instantiation
- Create a copy of `config_template.json` named `config.json`.
- Fill configuration details based on the used model (below).

### GPT-4 / GPT-3.5
- Adjust predefined `chatgpt`,  `chatgpt4` or create new configuration with an unique key.

| Key                 | Value                                                                                                                                                                                                                                                                                                                                                               |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model_id            | Model name based on [OpenAI model overview](https://platform.openai.com/docs/models/overview).                                                                                                                                                                                                                                                                      |
| prompt_token_cost   | Price per 1000 prompt tokens based on [OpenAI pricing](https://openai.com/pricing), used for calculating cumulative price per LLM instance.                                                                                                                                                                                                                         |
| response_token_cost | Price per 1000 response tokens based on [OpenAI pricing](https://openai.com/pricing), used for calculating cumulative price per LLM instance.                                                                                                                                                                                                                       |
| temperature         | Parameter of OpenAI models that controls randomness and the creativity of the responses (higher temperature = more diverse and unexpected responses). Value between 0.0 and 2.0, default is 1.0. More information can be found in the [OpenAI API reference](https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature).     |
| max_tokens          | The maximum number of tokens to generate in the chat completion. Value depends on the maximum context size of the model specified in the [OpenAI model overview](https://platform.openai.com/docs/models/overview). More information can be found in the [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create#chat/create-max_tokens). |
| stop                | String or array of strings specifying sequence of characters which if detected, stops further generation of tokens. More information can be found in the [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create#chat/create-stop).                                                                                                       |
| organization        | Organization to use for the API requests (may be empty).                                                                                                                                                                                                                                                                                                            |
| api_key             | Personal API key that will be used to access OpenAI API.                                                                                                                                                                                                                                                                                                            |

- Instantiate the language model based on the selected configuration key (predefined / custom).
```
lm = controller.ChatGPT(
    "path/to/config.json", 
    model_name=<configuration key>
)
```

### Llama-2
- Requires local hardware to run inference and a HuggingFace account.
- Adjust predefined `llama7b-hf`, `llama13b-hf`, `llama70b-hf` or create a new configuration with an unique key.

| Key                 | Value                                                                                                                                                                           |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model_id            | Specifies HuggingFace Llama 2 model identifier (`meta-llama/<model_id>`).                                                                                                       |
| cache_dir           | Local directory where model will be downloaded and accessed.                                                                                                                    |
| prompt_token_cost   | Price per 1000 prompt tokens (currently not used - local model = no cost).                                                                                                      |
| response_token_cost | Price per 1000 response tokens (currently not used - local model = no cost).                                                                                                    |
| temperature         | Parameter that controls randomness and the creativity of the responses (higher temperature = more diverse and unexpected responses). Value between 0.0 and 1.0, default is 0.6. |
| top_k               | Top-K sampling method described in [Transformers tutorial](https://huggingface.co/blog/how-to-generate). Default value is set to 10.                                            |
| max_tokens          | The maximum number of tokens to generate in the chat completion. More tokens require more memory.                                                                               |

- Instantiate the language model based on the selected configuration key (predefined / custom).
```
lm = controller.Llama2HF(
    "path/to/config.json", 
    model_name=<configuration key>
)
```
- Request access to Llama-2 via the [Meta form](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) using the same email address as for the HuggingFace account.
- After the access is granted, go to [HuggingFace Llama-2 model card](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), log in and accept the license (_"You have been granted access to this model"_ message should appear).
- Generate HuggingFace access token.
- Log in from CLI with: `huggingface-cli login --token <your token>`.

Note: 4-bit quantization is used to reduce the model size for inference. During instantiation, the model is downloaded from HuggingFace into the cache directory specified in the `config.json`. Running queries using larger models will require multiple GPUs (splitting across many GPUs is done automatically by the Transformers library).

## Adding LLMs
More LLMs can be added by following these steps:
- Create new class as a subclass of `AbstractLanguageModel`.
- Use the constructor for loading configuration and instantiating the language model (if needed). 
```
class CustomLanguageModel(AbstractLanguageModel):
    def __init__(
        self,
        config_path: str = "",
        model_name: str = "llama7b-hf",
        cache: bool = False
    ) -> None:
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        
        # Load data from configuration into variables if needed

        # Instantiate LLM if needed
```
- Implement `query` abstract method that is used to get a list of responses from the LLM (call to remote API or local model inference).
```
def query(self, query: str, num_responses: int = 1) -> Any:
    # Support caching 
    # Call LLM and retrieve list of responses - based on num_responses    
    # Return LLM response structure (not only raw strings)    
```
- Implement `get_response_texts` abstract method that is used to get a list of raw texts from the LLM response structure produced by `query`.
```
def get_response_texts(self, query_response: Union[List[Dict], Dict]) -> List[str]:
    # Retrieve list of raw strings from the LLM response structure    
```
