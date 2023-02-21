# Integrations

Besides the installation of this python package, you will also need to install packages and set environment variables depending on which chains you want to use.

Note: the reason these packages are not included in the dependencies by default is that as we imagine scaling this package, we do not want to force dependencies that are not needed.

The following use cases require specific installs and api keys:

- _OpenAI_:
  - Install requirements with `pip install openai`
  - Get an OpenAI api key and either set it as an environment variable (`OPENAI_API_KEY`) or pass it to the LLM constructor as `openai_api_key`.
- _Cohere_:
  - Install requirements with `pip install cohere`
  - Get a Cohere api key and either set it as an environment variable (`COHERE_API_KEY`) or pass it to the LLM constructor as `cohere_api_key`.
- _GooseAI_:
  - Install requirements with `pip install openai`
  - Get an GooseAI api key and either set it as an environment variable (`GOOSEAI_API_KEY`) or pass it to the LLM constructor as `gooseai_api_key`.
- _Hugging Face Hub_
  - Install requirements with `pip install huggingface_hub`
  - Get a Hugging Face Hub api token and either set it as an environment variable (`HUGGINGFACEHUB_API_TOKEN`) or pass it to the LLM constructor as `huggingfacehub_api_token`.
- _Petals_:
  - Install requirements with `pip install petals`
  - Get an GooseAI api key and either set it as an environment variable (`HUGGINGFACE_API_KEY`) or pass it to the LLM constructor as `huggingface_api_key`.
- _CerebriumAI_:
  - Install requirements with `pip install cerebrium`
  - Get a Cerebrium api key and either set it as an environment variable (`CEREBRIUMAI_API_KEY`) or pass it to the LLM constructor as `cerebriumai_api_key`.
- _PromptLayer_:
  - Install requirements with `pip install promptlayer` (be sure to be on version 0.1.62 or higher)
  - Get an API key from [promptlayer.com](http://www.promptlayer.com) and set it using `promptlayer.api_key=<API KEY>`
- _SerpAPI_:
  - Install requirements with `pip install google-search-results`
  - Get a SerpAPI api key and either set it as an environment variable (`SERPAPI_API_KEY`) or pass it to the LLM constructor as `serpapi_api_key`.
- _GoogleSearchAPI_:
  - Install requirements with `pip install google-api-python-client`
  - Get a Google api key and either set it as an environment variable (`GOOGLE_API_KEY`) or pass it to the LLM constructor as `google_api_key`. You will also need to set the `GOOGLE_CSE_ID` environment variable to your custom search engine id. You can pass it to the LLM constructor as `google_cse_id` as well.
- _WolframAlphaAPI_:
  - Install requirements with `pip install wolframalpha`
  - Get a Wolfram Alpha api key and either set it as an environment variable (`WOLFRAM_ALPHA_APPID`) or pass it to the LLM constructor as `wolfram_alpha_appid`.
- _NatBot_:
  - Install requirements with `pip install playwright`
- _Wikipedia_:
  - Install requirements with `pip install wikipedia`
- _Elasticsearch_:
  - Install requirements with `pip install elasticsearch`
  - Set up Elasticsearch backend. If you want to do locally, [this](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/getting-started.html) is a good guide.
- _FAISS_:
  - Install requirements with `pip install faiss` for Python 3.7 and `pip install faiss-cpu` for Python 3.10+.
- _Manifest_:
  - Install requirements with `pip install manifest-ml` (Note: this is only available in Python 3.8+ currently).
- _OpenSearch_:
  - Install requirements with `pip install opensearch-py`
  - If you want to set up OpenSearch on your local, [here](https://opensearch.org/docs/latest/)


If you are using the `NLTKTextSplitter` or the `SpacyTextSplitter`, you will also need to install the appropriate models. For example, if you want to use the `SpacyTextSplitter`, you will need to install the `en_core_web_sm` model with `python -m spacy download en_core_web_sm`. Similarly, if you want to use the `NLTKTextSplitter`, you will need to install the `punkt` model with `python -m nltk.downloader punkt`.
