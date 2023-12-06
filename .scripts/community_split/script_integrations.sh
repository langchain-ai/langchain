#!/bin/bash

cd libs

git checkout master -- langchain/{langchain,tests}
git checkout master -- core/{langchain_core,tests}
git checkout master -- experimental/{langchain_experimental,tests}
rm -rf community/{langchain_community,tests}
rm -rf partners/openai/{langchain_openai,tests}

mkdir -p community/langchain_community
touch community/langchain_community/__init__.py
touch community/README.md
mkdir -p community/tests
touch community/tests/__init__.py
mkdir community/tests/unit_tests
touch community/tests/unit_tests/__init__.py
mkdir community/tests/integration_tests/
touch community/tests/integration_tests/__init__.py
mkdir -p community/langchain_community/utils
touch community/langchain_community/utils/__init__.py
mkdir -p community/tests/unit_tests/utils
touch community/tests/unit_tests/utils/__init__.py
mkdir -p community/langchain_community/indexes
touch community/langchain_community/indexes/__init__.py
mkdir community/tests/unit_tests/indexes
touch community/tests/unit_tests/indexes/__init__.py


mkdir -p partners/openai/langchain_openai/chat_models
mkdir -p partners/openai/langchain_openai/llms
mkdir -p partners/openai/langchain_openai/embeddings
touch partners/openai/langchain_openai/__init__.py
touch partners/openai/README.md
touch partners/openai/langchain_openai/{llms,chat_models,embeddings}/__init__.py

cd langchain

git grep -l 'from langchain.pydantic_v1' | xargs sed -i '' 's/from langchain.pydantic_v1/from langchain_core.pydantic_v1/g'
git grep -l 'from langchain.tools.base' | xargs sed -i '' 's/from langchain.tools.base/from langchain_core.tools/g'
git grep -l 'from langchain.chat_models.base' | xargs sed -i '' 's/from langchain.chat_models.base/from langchain_core.language_models.chat_models/g'
git grep -l 'from langchain.llms.base' | xargs sed -i '' 's/from langchain.llms.base/from langchain_core.language_models.llms/g'
git grep -l 'from langchain.embeddings.base' | xargs sed -i '' 's/from langchain.embeddings.base/from langchain_core.embeddings/g'
git grep -l 'from langchain.vectorstores.base' | xargs sed -i '' 's/from langchain.vectorstores.base/from langchain_core.vectorstores/g'
git grep -l 'from langchain.agents.tools' | xargs sed -i '' 's/from langchain.agents.tools/from langchain_core.tools/g'
git grep -l 'from langchain.schema.output' | xargs sed -i '' 's/from langchain.schema.output/from langchain_core.outputs/g'
git grep -l 'from langchain.schema.messages' | xargs sed -i '' 's/from langchain.schema.messages/from langchain_core.messages/g'
git grep -l 'from langchain.schema.embeddings' | xargs sed -i '' 's/from langchain.schema.embeddings/from langchain_core.embeddings/g'

cd ..

mv langchain/langchain/chat_loaders community/langchain_community
mv langchain/langchain/document_loaders community/langchain_community
mv langchain/langchain/docstore community/langchain_community
mv langchain/langchain/document_transformers community/langchain_community
mv langchain/langchain/embeddings community/langchain_community
mv langchain/langchain/graphs community/langchain_community
mv langchain/langchain/llms community/langchain_community
mv langchain/langchain/chat_models community/langchain_community
mv langchain/langchain/memory/chat_message_histories community/langchain_community
mv langchain/langchain/storage community/langchain_community
mv langchain/langchain/tools community/langchain_community
mv langchain/langchain/utilities community/langchain_community
mv langchain/langchain/vectorstores community/langchain_community
mv langchain/langchain/adapters community/langchain_community
mv langchain/langchain/agents/agent_toolkits community/langchain_community
mv langchain/langchain/cache.py community/langchain_community
mv langchain/langchain/callbacks community/langchain_community/callbacks
mv langchain/langchain/indexes/base.py community/langchain_community/indexes
mv langchain/langchain/indexes/_sql_record_manager.py community/langchain_community/indexes
mv langchain/langchain/utils/math.py community/langchain_community/utils

mv langchain/langchain/utils/openai.py partners/openai/langchain_openai/utils.py
mv langchain/langchain/utils/openai_functions.py partners/openai/langchain_openai/functions.py

mv langchain/langchain/utils/json_schema.py core/langchain_core/utils
mv langchain/langchain/utils/html.py core/langchain_core/utils
mv langchain/langchain/utils/strings.py core/langchain_core/utils
cat langchain/langchain/utils/env.py >> core/langchain_core/utils/env.py
rm langchain/langchain/utils/env.py

mv langchain/tests/unit_tests/chat_loaders community/tests/unit_tests
mv langchain/tests/unit_tests/document_loaders community/tests/unit_tests
mv langchain/tests/unit_tests/docstore community/tests/unit_tests
mv langchain/tests/unit_tests/document_transformers community/tests/unit_tests
mv langchain/tests/unit_tests/embeddings community/tests/unit_tests
mv langchain/tests/unit_tests/graphs community/tests/unit_tests
mv langchain/tests/unit_tests/llms community/tests/unit_tests
mv langchain/tests/unit_tests/chat_models community/tests/unit_tests
mv langchain/tests/unit_tests/memory/chat_message_histories community/tests/unit_tests
mv langchain/tests/unit_tests/storage community/tests/unit_tests
mv langchain/tests/unit_tests/tools community/tests/unit_tests
mv langchain/tests/unit_tests/utilities community/tests/unit_tests
mv langchain/tests/unit_tests/vectorstores community/tests/unit_tests
mv langchain/tests/unit_tests/callbacks community/tests/unit_tests
mv langchain/tests/unit_tests/indexes/test_sql_record_manager.py community/tests/unit_tests/indexes
mv langchain/tests/unit_tests/utils/test_math.py community/tests/unit_tests/utils

mkdir -p langchain/tests/unit_tests/llms
cp {community,langchain}/tests/unit_tests/llms/fake_llm.py
cp {community,langchain}/tests/unit_tests/llms/fake_chat_model.py
mkdir -p langchain/tests/unit_tests/callbacks
cp {community,langchain}/tests/unit_tests/callbacks/fake_callback_handler.py

mv langchain/tests/unit_tests/utils/test_json_schema.py core/tests/unit_tests/utils
mv langchain/tests/unit_tests/utils/test_html.py core/tests/unit_tests/utils

mv langchain/tests/integration_tests/document_loaders community/tests/integration_tests
mv langchain/tests/integration_tests/embeddings community/tests/integration_tests
mv langchain/tests/integration_tests/graphs community/tests/integration_tests
mv langchain/tests/integration_tests/llms community/tests/integration_tests
mv langchain/tests/integration_tests/chat_models community/tests/integration_tests
mv langchain/tests/integration_tests/memory/chat_message_histories community/tests/integration_tests
mv langchain/tests/integration_tests/storage community/tests/integration_tests
mv langchain/tests/integration_tests/tools community/tests/integration_tests
mv langchain/tests/integration_tests/utilities community/tests/integration_tests
mv langchain/tests/integration_tests/vectorstores community/tests/integration_tests
mv langchain/tests/integration_tests/adapters community/tests/integration_tests
mv langchain/tests/integration_tests/callbacks community/tests/integration_tests

git grep -l 'from langchain.utils.json_schema' | xargs sed -i '' 's/from langchain.utils.json_schema/from langchain_core.utils.json_schema/g'
git grep -l 'from langchain.utils.html' | xargs sed -i '' 's/from langchain.utils.html/from langchain_core.utils.html/g'
git grep -l 'from langchain.utils.strings' | xargs sed -i '' 's/from langchain.utils.strings/from langchain_core.utils.strings/g'
git grep -l 'from langchain.utils.env' | xargs sed -i '' 's/from langchain.utils.env/from langchain_core.utils.env/g'

git add community
cd community

git grep -l 'from langchain.pydantic_v1' | xargs sed -i '' 's/from langchain.pydantic_v1/from langchain_core.pydantic_v1/g'
git grep -l 'from langchain.callbacks.base' | xargs sed -i '' 's/from langchain.callbacks.base/from langchain_core.callbacks/g'
git grep -l 'from langchain.callbacks.stdout' | xargs sed -i '' 's/from langchain.callbacks.stdout/from langchain_core.callbacks/g'
git grep -l 'from langchain.callbacks.streaming_stdout' | xargs sed -i '' 's/from langchain.callbacks.streaming_stdout/from langchain_core.callbacks/g'
git grep -l 'from langchain.callbacks.manager' | xargs sed -i '' 's/from langchain.callbacks.manager/from langchain_core.callbacks/g'
git grep -l 'from langchain.tools.base' | xargs sed -i '' 's/from langchain.tools.base/from langchain_core.tools/g'
git grep -l 'from langchain.agents.tools' | xargs sed -i '' 's/from langchain.agents.tools/from langchain_core.tools/g'
git grep -l 'from langchain.schema.output' | xargs sed -i '' 's/from langchain.schema.output/from langchain_core.outputs/g'
git grep -l 'from langchain.schema.messages' | xargs sed -i '' 's/from langchain.schema.messages/from langchain_core.messages/g'

git grep -l 'from langchain.utils.math' | xargs sed -i '' 's/from langchain.utils.math/from langchain_community.utils.math/g'
git grep -l 'from langchain.utils.openai_functions' | xargs sed -i '' 's/from langchain.utils.openai_functions/from langchain_openai.functions/g'
git grep -l 'from langchain.utils.openai' | xargs sed -i '' 's/from langchain.utils.openai/from langchain_openai.utils/g'
git grep -l 'from langchain.chat_models.openai' | xargs sed -i '' 's/from langchain.chat_models.openai/from langchain_openai.chat_models/g'
git grep -l 'from langchain.chat_models.azure_openai' | xargs sed -i '' 's/from langchain.chat_models.azure_openai/from langchain_openai.chat_models/g'
git grep -l 'from langchain.embeddings.openai' | xargs sed -i '' 's/from langchain.embeddings.openai/from langchain_openai.embeddings/g'
git grep -l 'from langchain.embeddings.azure_openai' | xargs sed -i '' 's/from langchain.embeddings.azure_openai/from langchain_openai.embeddings/g'
git grep -l 'from langchain.adapters.openai' | xargs sed -i '' 's/from langchain.adapters.openai/from langchain_openai.adapters/g'
git grep -l 'from langchain.llms.openai' | xargs sed -i '' 's/from langchain.llms.openai/from langchain_openai.llms/g'
git grep -l 'from langchain.utils' | xargs sed -i '' 's/from langchain.utils/from langchain_core.utils/g'
git grep -l 'from langchain\.' | xargs sed -i '' 's/from langchain\./from langchain_community./g'
git grep -l 'from langchain_community.memory.chat_message_histories' | xargs sed -i '' 's/from langchain_community.memory.chat_message_histories/from langchain_community.chat_message_histories/g'
git grep -l 'from langchain_community.agents.agent_toolkits' | xargs sed -i '' 's/from langchain_community.agents.agent_toolkits/from langchain_community.agent_toolkits/g'

git grep -l 'from langchain_community\.text_splitter' | xargs sed -i '' 's/from langchain_community\.text_splitter/from langchain.text_splitter/g'
git grep -l 'from langchain_community\.chains' | xargs sed -i '' 's/from langchain_community\.chains/from langchain.chains/g'
git grep -l 'from langchain_community\.agents' | xargs sed -i '' 's/from langchain_community\.agents/from langchain.agents/g'
git grep -l 'from langchain_community\.memory' | xargs sed -i '' 's/from langchain_community\.memory/from langchain.memory/g'

cd ..
git checkout master -- langchain/langchain
cd langchain

python ../../.scripts/community_split/update_imports.py langchain/chat_loaders langchain_community.chat_loaders
python ../../.scripts/community_split/update_imports.py langchain/callbacks langchain_community.callbacks
python ../../.scripts/community_split/update_imports.py langchain/document_loaders langchain_community.document_loaders
python ../../.scripts/community_split/update_imports.py langchain/docstore langchain_community.docstore
python ../../.scripts/community_split/update_imports.py langchain/document_transformers langchain_community.document_transformers
python ../../.scripts/community_split/update_imports.py langchain/embeddings langchain_community.embeddings
python ../../.scripts/community_split/update_imports.py langchain/graphs langchain_community.graphs
python ../../.scripts/community_split/update_imports.py langchain/llms langchain_community.llms
python ../../.scripts/community_split/update_imports.py langchain/chat_models langchain_community.chat_models
python ../../.scripts/community_split/update_imports.py langchain/memory/chat_message_histories langchain_community.chat_message_histories
python ../../.scripts/community_split/update_imports.py langchain/storage langchain_community.storage
python ../../.scripts/community_split/update_imports.py langchain/tools langchain_community.tools
python ../../.scripts/community_split/update_imports.py langchain/utilities langchain_community.utilities
python ../../.scripts/community_split/update_imports.py langchain/vectorstores langchain_community.vectorstores
python ../../.scripts/community_split/update_imports.py langchain/adapters langchain_community.adapters
python ../../.scripts/community_split/update_imports.py langchain/agents/agent_toolkits langchain_community.agent_toolkits
python ../../.scripts/community_split/update_imports.py langchain/cache.py langchain_community.cache
python ../../.scripts/community_split/update_imports.py langchain/utils/math.py langchain_community.utils.math
python ../../.scripts/community_split/update_imports.py langchain/utils/json_schema.py langchain_core.utils.json_schema
python ../../.scripts/community_split/update_imports.py langchain/utils/html.py langchain_core.utils.html
python ../../.scripts/community_split/update_imports.py langchain/utils/env.py langchain_core.utils.env
python ../../.scripts/community_split/update_imports.py langchain/utils/strings.py langchain_core.utils.strings
python ../../.scripts/community_split/update_imports.py langchain/utils/openai.py langchain_openai.utils
python ../../.scripts/community_split/update_imports.py langchain/utils/openai_functions.py langchain_openai.functions

git grep -l 'from langchain.llms.base ' | xargs sed -i '' 's/from langchain.llms.base /from langchain_core.language_models.llms /g'
git grep -l 'from langchain.chat_models.base ' | xargs sed -i '' 's/from langchain.chat_models.base /from langchain_core.language_models.chat_models /g'
git grep -l 'from langchain.tools.base' | xargs sed -i '' 's/from langchain.tools.base/from langchain_core.tools/g'
git grep -l 'from langchain_community.llms.openai' | xargs sed -i '' 's/from langchain_community.llms.openai/from langchain_openai.llms/g'
git grep -l 'from langchain_community.chat_models.openai' | xargs sed -i '' 's/from langchain_community.chat_models.openai/from langchain_openai.chat_models/g'
git grep -l 'from langchain_community.chat_models.azure_openai' | xargs sed -i '' 's/from langchain_community.chat_models.azure_openai/from langchain_openai.chat_models/g'
git grep -l 'from langchain_community.embeddings.openai' | xargs sed -i '' 's/from langchain_community.embeddings.openai/from langchain_openai.embeddings/g'
git grep -l 'from langchain_community.embeddings.azure_openai' | xargs sed -i '' 's/from langchain_community.embeddings.azure_openai/from langchain_openai.embeddings/g'

cd ..

mv community/langchain_community/utilities/loading.py langchain/langchain/utilities
mv community/langchain_community/utilities/asyncio.py langchain/langchain/utilities

mv community/langchain_community/chat_models/openai.py partners/openai/langchain_openai/chat_models/base.py
mv community/langchain_community/chat_models/azure_openai.py partners/openai/langchain_openai/chat_models/azure.py
mv community/langchain_community/llms/openai.py partners/openai/langchain_openai/llms/base.py
mv community/langchain_community/embeddings/openai.py partners/openai/langchain_openai/embeddings/base.py
mv community/langchain_community/embeddings/azure_openai.py partners/openai/langchain_openai/embeddings/azure.py
mv community/langchain_community/adapters/openai.py partners/openai/langchain_openai/adapters.py

git add partners core

rm community/langchain_community/chat_models/base.py
rm community/langchain_community/llms/base.py
rm community/langchain_community/tools/base.py
rm community/langchain_community/embeddings/base.py
rm community/langchain_community/vectorstores/base.py
rm community/langchain_community/callbacks/{base,stdout,streaming_stdout}.py
rm community/langchain_community/callbacks/tracers/{base,evaluation,langchain,langchain_v1,log_stream,root_listeners,run_collector,schemas,stdout}.py

git checkout master -- langchain/tests/unit_tests/chat_models/test_base.py
git checkout master -- langchain/tests/unit_tests/llms/test_base.py
git checkout master -- langchain/tests/unit_tests/tools/test_base.py
git checkout master -- langchain/tests/unit_tests/schema
touch langchain/tests/unit_tests/{llms,chat_models,tools,callbacks,runnables}/__init__.py

cp core/Makefile community
cp core/Makefile partners/openai
sed -i '' 's/libs\/core/libs\/community/g' community/Makefile
sed -i '' 's/libs\/core/libs\/partners\/openai/g' partners/openai/Makefile
cp -r core/scripts community
cp -r core/scripts partners/openai

printf 'from langchain_openai.llms import BaseOpenAI, OpenAI, AzureOpenAI\nfrom langchain_openai.chat_models import _import_tiktoken, ChatOpenAI, AzureChatOpenAI\nfrom langchain_openai.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings\nfrom langchain_openai.functions import convert_pydantic_to_openai_function, convert_pydantic_to_openai_tool\n\n__all__ = ["_import_tiktoken", "OpenAI", "AzureOpenAI", "ChatOpenAI", "AzureChatOpenAI", "OpenAIEmbeddings", "AzureOpenAIEmbeddings", "convert_pydantic_to_openai_function", "convert_pydantic_to_openai_tool", "BaseOpenAI"]' >> partners/openai/langchain_openai/__init__.py
printf 'from langchain_openai.llms.base import update_token_usage, _stream_response_to_generation_chunk, _update_response, _streaming_response_template, _create_retry_decorator, completion_with_retry, acompletion_with_retry, BaseOpenAI, OpenAIChat, OpenAI, AzureOpenAI\n\n__all__ = ["update_token_usage", "_stream_response_to_generation_chunk", "_update_response", "_streaming_response_template", "_create_retry_decorator", "completion_with_retry", "acompletion_with_rety", "OpenAIChat", "OpenAI", "AzureOpenAI", "BaseOpenAI"]' >> partners/openai/langchain_openai/llms/__init__.py
printf 'from langchain_openai.chat_models.base import _create_retry_decorator, acompletion_with_retry, _convert_delta_to_message_chunk, _import_tiktoken, ChatOpenAI\nfrom langchain_openai.chat_models.azure import AzureChatOpenAI\n\n__all__ = ["_create_retry_decorator", "acompletion_with_retry", "_convert_delta_to_message_chunk", "_import_tiktoken", "ChatOpenAI", "AzureChatOpenAI"]' >> partners/openai/langchain_openai/chat_models/__init__.py
printf 'from langchain_openai.embeddings.base import _create_retry_decorator, _async_retry_decorator, _check_response, embed_with_retry, async_embed_with_retry, _is_openai_v1, OpenAIEmbeddings\nfrom langchain_openai.embeddings.azure import AzureOpenAIEmbeddings\n\n\n__all__ = ["_create_retry_decorator", "_async_retry_decorator", "_check_response", "embed_with_retry", "async_embed_with_retry", "_is_openai_v1", "OpenAIEmbeddings", "AzureOpenAIEmbeddings"]' >> partners/openai/langchain_openai/embeddings/__init__.py


sed -i '' 's/from\ langchain_openai.chat_models\ /from\ langchain_openai.chat_models.base\ /g' partners/openai/langchain_openai/chat_models/azure.py
sed -i '' 's/from\ langchain_openai.embeddings\ /from\ langchain_openai.embeddings.base\ /g' partners/openai/langchain_openai/embeddings/azure.py

echo '"""
**Utility functions** for LangChain.

These functions do not depend on any other LangChain module.
"""

from langchain_core.utils.formatting import StrictFormatter, formatter
from langchain_core.utils.input import (
    get_bolded_text,
    get_color_mapping,
    get_colored_text,
    print_text,
)
from langchain_core.utils.loading import try_load_from_hub
from langchain_core.utils.utils import (
    build_extra_kwargs,
    check_package_version,
    convert_to_secret_str,
    get_pydantic_field_names,
    guard_import,
    mock_now,
    raise_for_status_with_text,
    xor_args,
)
from langchain_core.utils.env import get_from_env, get_from_dict_or_env
from langchain_core.utils.strings import stringify_dict, comma_list, stringify_value

__all__ = [
    "StrictFormatter",
    "check_package_version",
    "convert_to_secret_str",
    "formatter",
    "get_bolded_text",
    "get_color_mapping",
    "get_colored_text",
    "get_pydantic_field_names",
    "guard_import",
    "mock_now",
    "print_text",
    "raise_for_status_with_text",
    "xor_args",
    "try_load_from_hub",
    "build_extra_kwargs",
    "get_from_env",
    "get_from_dict_or_env",
    "stringify_dict",
    "comma_list",
    "stringify_value",
]
' > core/langchain_core/utils/__init__.py

cd core
make format
cd ../langchain
make format
cd ../experimental
make format
cd ../community
make format
cd ../partners/openai
make format

cd ../..
sed -E -i '' '1 s/(.*)/\1\ \ \#\ noqa\:\ E501/g' langchain/langchain/agents/agent_toolkits/conversational_retrieval/openai_functions.py
sed -i '' 's/llms\.loading\.get_type_to_cls_dict/llms.get_type_to_cls_dict/g' langchain/tests/unit_tests/chains/test_llm.py