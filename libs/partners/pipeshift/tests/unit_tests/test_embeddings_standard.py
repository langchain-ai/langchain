# """Standard LangChain interface tests"""

# from typing import Tuple, Type

# from langchain_core.embeddings import Embeddings
# from langchain_standard_tests.unit_tests.embeddings import EmbeddingsUnitTests

# from langchain_pipeshift import PipeshiftEmbeddings


# class TestPipeshiftStandard(EmbeddingsUnitTests):
#     @property
#     def embeddings_class(self) -> Type[Embeddings]:
#         return PipeshiftEmbeddings

#     @property
#     def embeddings_params(self) -> dict:
#         return {"model": "meta-llama/Llama-2-7b-chat-hf"}

#     @property
#     def init_from_env_params(self) -> Tuple[dict, dict, dict]:
#         return (
#             {
#                 "PIPESHIFT_API_KEY": "pipeshift_api_secret",
#                 "PIPESHIFT_API_BASE": "https://api.baseurl.com",
#             },
#             {},
#             {
#                 "pipeshift_api_key": "pipeshift_api_secret",
#                 "pipeshift_api_base": "https://api.baseurl.com",
#             },
#         )
