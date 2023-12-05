#!/bin/bash

cd libs

git checkout master -- langchain/{langchain,tests}
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

mv langchain/langchain/utils/json_schema.py core/langchain_core/utils

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

mv langchain/tests/unit_tests/utils/test_json_schema.py core/tests/unit_tests/utils

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

git add community
cd community

git grep -l 'from langchain.pydantic_v1' | xargs sed -i '' 's/from langchain.pydantic_v1/from langchain_core.pydantic_v1/g'
git grep -l 'from langchain.utils.math' | xargs sed -i '' 's/from langchain.utils.math/from langchain_community.utils.math/g'
git grep -l 'from langchain.utils.json_schema' | xargs sed -i '' 's/from langchain.utils.json_schema/from langchain_core.utils.json_schema/g'
git grep -l 'from langchain.utils.openai_functions' | xargs sed -i '' 's/from langchain.utils.openai_functions/from langchain_openai.functions/g'
git grep -l 'from langchain.utils.openai' | xargs sed -i '' 's/from langchain.utils.openai/from langchain_openai.utils/g'
git grep -l 'from langchain.chat_models.openai' | xargs sed -i '' 's/from langchain.chat_models.openai/from langchain_openai.chat_model/g'
git grep -l 'from langchain.embeddings.openai' | xargs sed -i '' 's/from langchain.embeddings.openai/from langchain_openai.embedding/g'
git grep -l 'from langchain.llms.openai' | xargs sed -i '' 's/from langchain.llms.openai/from langchain_openai.llm/g'
git grep -l 'from langchain.tools.base' | xargs sed -i '' 's/from langchain.tools.base/from langchain_core.tools/g'
git grep -l 'from langchain.agents.tools' | xargs sed -i '' 's/from langchain.agents.tools/from langchain_core.tools/g'
git grep -l 'from langchain.callbacks' | xargs sed -i '' 's/from langchain.callbacks/from langchain_core.callbacks/g'
git grep -l 'from langchain.schema.output' | xargs sed -i '' 's/from langchain.schema.output/from langchain_core.outputs/g'
git grep -l 'from langchain.schema.messages' | xargs sed -i '' 's/from langchain.schema.messages/from langchain_core.messages/g'
git grep -l 'from langchain.utils' | xargs sed -i '' 's/from langchain.utils/from langchain_core.utils/g'
git grep -l 'from langchain\.' | xargs sed -i '' 's/from langchain\./from langchain_community./g'
git grep -l 'from langchain_community.memory.chat_message_histories' | xargs sed -i '' 's/from langchain_community.memory.chat_message_histories/from langchain_community.chat_message_histories/g'
git grep -l 'from langchain_community.agents.agent_toolkits' | xargs sed -i '' 's/from langchain_community.agents.agent_toolkits/from langchain_community.agent_toolkits/g'
git grep -l 'from langchain.cache' | xargs sed -i '' 's/from langchain.cache/from langchain_community.cache/g'

cd ..
git checkout master -- langchain
cd langchain

python ../../.scripts/community_split/update_imports.py langchain/chat_loaders langchain_community.chat_loaders
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
python ../../.scripts/community_split/update_imports.py langchain/utils/json_schema.py langchain_community.utils.json_schema

git grep -l 'from langchain.llms.base ' | xargs sed -i '' 's/from langchain.llms.base /from langchain_core.language_models.llms /g'
git grep -l 'from langchain.chat_models.base ' | xargs sed -i '' 's/from langchain.chat_models.base /from langchain_core.language_models.chat_models /g'
git grep -l 'from langchain.tools.base' | xargs sed -i '' 's/from langchain.tools.base/from langchain_core.tools/g'
git grep -l 'from langchain_community.llms.openai' | xargs sed -i '' 's/from langchain_community.llms.openai/from langchain_openai.llm/g'
git grep -l 'from langchain_community.chat_models.openai' | xargs sed -i '' 's/from langchain_community.chat_models.openai/from langchain_openai.chat_model/g'
git grep -l 'from langchain_community.embeddings.openai' | xargs sed -i '' 's/from langchain_community.embeddings.openai/from langchain_openai.embedding/g'
git grep -l 'from langchain.utils.json_schema' | xargs sed -i '' 's/from langchain.utils.json_schema/from langchain_core.utils.json_schema/g'

cd ..

mkdir -p partners/openai/langchain_openai/chat_models
mkdir -p partners/openai/langchain_openai/llms
mkdir -p partners/openai/langchain_openai/embeddings
touch partners/openai/langchain_openai/__init__.py
touch partners/openai/README.md

mv community/langchain_community/utilities/loading.py langchain/langchain/utilities
mv community/langchain_community/utilities/asyncio.py langchain/langchain/utilities
mv community/langchain_community/utilities/requests.py langchain/langchain/utilities

mv community/langchain_community/chat_models/openai.py partners/openai/langchain_openai/chat_models/base.py
mv community/langchain_community/chat_models/azure_openai.py partners/openai/langchain_openai/chat_models/azure.py
mv community/langchain_community/llms/openai.py partners/openai/langchain_openai/llms/base.py
mv community/langchain_community/embeddings/openai.py partners/openai/langchain_openai/embeddings/base.py
mv community/langchain_community/embeddings/azure_openai.py partners/openai/langchain_openai/embeddings/azure.py
cp langchain/langchain/utils/openai.py partners/openai/langchain_openai/utils.py
cp langchain/langchain/utils/openai_functions.py partners/openai/langchain_openai/functions.py


git grep -l 'from langchain.utils.json_schema' | xargs sed -i '' 's/from langchain.utils.json_schema/from langchain_core.utils.json_schema/g'

git add partners core

rm community/langchain_community/chat_models/base.py
rm community/langchain_community/llms/base.py
rm community/langchain_community/tools/base.py
rm community/langchain_community/embeddings/base.py
rm community/langchain_community/vectorstores/base.py

git checkout master -- langchain/tests/unit_tests/chat_models/test_base.py
git checkout master -- langchain/tests/unit_tests/llms/test_base.py
git checkout master -- langchain/tests/unit_tests/tools/test_base.py

cp core/Makefile community
cp core/Makefile partners/openai
sed -i '' 's/libs\/core/libs\/community/g' community/Makefile
sed -i '' 's/libs\/core/libs\/partners\/openai/g' partners/openai/Makefile
cp -r core/scripts community
cp -r core/scripts partners/openai


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
