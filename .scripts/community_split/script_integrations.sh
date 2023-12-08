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


mkdir -p partners/openai/langchain_openai/{chat_models,llms,embeddings}
mkdir -p partners/openai/tests/{unit_tests,integration_tests}/llms
mkdir -p partners/openai/tests/{unit_tests,integration_tests}/chat_models
mkdir -p partners/openai/tests/{unit_tests,integration_tests}/embeddings
mkdir -p partners/openai/tests/integration_tests/adapters
touch partners/openai/langchain_openai/__init__.py
touch partners/openai/README.md
touch partners/openai/langchain_openai/{llms,chat_models,embeddings}/__init__.py
touch partners/openai/tests/{unit_tests,integration_tests}/__init__.py
touch partners/openai/tests/{unit_tests,integration_tests}/chat_models/__init__.py
touch partners/openai/tests/{unit_tests,integration_tests}/llms/__init__.py
touch partners/openai/tests/{unit_tests,integration_tests}/embeddings/__init__.py

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
mv langchain/tests/integration_tests/{test_kuzu,test_nebulagraph}.py community/tests/integration_tests/graphs
touch community/tests/integration_tests/{chat_message_histories,tools}/__init__.py


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
git grep -l 'from langchain.callbacks.tracers.base' | xargs sed -i '' 's/from langchain.callbacks.tracers.base/from langchain_core.tracers/g'
git grep -l 'from langchain.tools.base' | xargs sed -i '' 's/from langchain.tools.base/from langchain_core.tools/g'
git grep -l 'from langchain.agents.tools' | xargs sed -i '' 's/from langchain.agents.tools/from langchain_core.tools/g'
git grep -l 'from langchain.schema.output' | xargs sed -i '' 's/from langchain.schema.output/from langchain_core.outputs/g'
git grep -l 'from langchain.schema.messages' | xargs sed -i '' 's/from langchain.schema.messages/from langchain_core.messages/g'

git grep -l 'from langchain.utils.math' | xargs sed -i '' 's/from langchain.utils.math/from langchain_community.utils.math/g'
git grep -l 'from langchain.utils.openai_functions' | xargs sed -i '' 's/from langchain.utils.openai_functions/from langchain_openai.functions/g'
git grep -l 'from langchain.utils.openai' | xargs sed -i '' 's/from langchain.utils.openai/from langchain_openai.utils/g'
git grep -l 'from langchain.chat_models.openai' | xargs sed -i '' 's/from\ langchain.chat_models.openai/from langchain_openai.chat_models/g'
git grep -l 'from langchain.chat_models import ChatOpenAI' | xargs sed -i '' 's/from\ langchain.chat_models\ import\ ChatOpenAI/from langchain_openai.chat_models import ChatOpenAI/g'
git grep -l 'from langchain.chat_models import AzureChatOpenAI' | xargs sed -i '' 's/from\ langchain.chat_models\ import\ AzureChatOpenAI/from langchain_openai.chat_models import AzureChatOpenAI/g'
git grep -l 'from langchain.llms import OpenAI' | xargs sed -i '' 's/from\ langchain.llms\ import\ OpenAI/from langchain_openai.llms import OpenAI/g'
git grep -l 'from langchain.llms import AzureOpenAI' | xargs sed -i '' 's/from\ langchain.llms\ import\ AzureOpenAI/from langchain_openai.llms import AzureOpenAI/g'
git grep -l 'from langchain.embeddings import OpenAIEmbeddings' | xargs sed -i '' 's/from\ langchain.embeddings\ import\ OpenAIEmbeddings/from langchain_openai.embeddings import OpenAIEmbeddings/g'
git grep -l 'from langchain.embeddings import AzureOpenAIEmbeddings' | xargs sed -i '' 's/from\ langchain.embeddings\ import\ AzureOpenAIEmbeddings/from langchain_openai.embeddings import AzureOpenAIEmbeddings/g'
git grep -l 'from langchain.chat_models.azure_openai' | xargs sed -i '' 's/from\ langchain.chat_models.azure_openai/from langchain_openai.chat_models/g'
git grep -l 'from langchain.embeddings.openai' | xargs sed -i '' 's/from langchain.embeddings.openai/from langchain_openai.embeddings/g'
git grep -l 'from langchain.embeddings.azure_openai' | xargs sed -i '' 's/from langchain.embeddings.azure_openai/from langchain_openai.embeddings/g'
git grep -l 'from langchain.adapters.openai' | xargs sed -i '' 's/from langchain.adapters.openai/from langchain_openai.adapters/g'
git grep -l 'from langchain.adapters import openai' | xargs sed -i '' 's/from\ langchain.adapters\ import\ openai/from\ langchain_openai\ import\ adapters/g'
git grep -l 'from langchain.llms.openai' | xargs sed -i '' 's/from langchain.llms.openai/from langchain_openai.llms/g'
git grep -l 'from langchain.utils' | xargs sed -i '' 's/from langchain.utils/from langchain_core.utils/g'
git grep -l 'from langchain\.' | xargs sed -i '' 's/from langchain\./from langchain_community./g'
git grep -l 'from langchain_community.memory.chat_message_histories' | xargs sed -i '' 's/from langchain_community.memory.chat_message_histories/from langchain_community.chat_message_histories/g'
git grep -l 'from langchain_community.agents.agent_toolkits' | xargs sed -i '' 's/from langchain_community.agents.agent_toolkits/from langchain_community.agent_toolkits/g'

git grep -l 'from langchain_community\.text_splitter' | xargs sed -i '' 's/from langchain_community\.text_splitter/from langchain.text_splitter/g'
git grep -l 'from langchain_community\.chains' | xargs sed -i '' 's/from langchain_community\.chains/from langchain.chains/g'
git grep -l 'from langchain_community\.agents' | xargs sed -i '' 's/from langchain_community\.agents/from langchain.agents/g'
git grep -l 'from langchain_community\.memory' | xargs sed -i '' 's/from langchain_community\.memory/from langchain.memory/g'
git grep -l 'langchain\.__version__' | xargs sed -i '' 's/langchain\.__version__/langchain_community.__version__/g'
git grep -l 'langchain\.document_loaders' | xargs sed -i '' 's/langchain\.document_loaders/langchain_community.document_loaders/g'
git grep -l 'langchain\.callbacks' | xargs sed -i '' 's/langchain\.callbacks/langchain_community.callbacks/g'
git grep -l 'langchain\.tools' | xargs sed -i '' 's/langchain\.tools/langchain_community.tools/g'
git grep -l 'langchain\.llms' | xargs sed -i '' 's/langchain\.llms/langchain_community.llms/g'
git grep -l 'import langchain$' | xargs sed -i '' 's/import\ langchain$/import\ langchain_community/g'
git grep -l 'from\ langchain\ ' | xargs sed -i '' 's/from\ langchain\ /from\ langchain_community\ /g'
git grep -l 'langchain_core.language_models.llmsten' | xargs sed -i '' 's/langchain_core.language_models.llmsten/langchain_community.llms.baseten/g'


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

git grep -l 'langchain_core.language_models.llmsten' | xargs sed -i '' 's/langchain_core.language_models.llmsten/langchain_community.llms.baseten/g'

cd ..

mv community/langchain_community/utilities/loading.py langchain/langchain/utilities
mv community/langchain_community/utilities/asyncio.py langchain/langchain/utilities

mv community/langchain_community/chat_models/openai.py partners/openai/langchain_openai/chat_models/base.py
mv community/langchain_community/chat_models/azure_openai.py partners/openai/langchain_openai/chat_models/azure.py
mv community/langchain_community/llms/openai.py partners/openai/langchain_openai/llms/base.py
mv community/langchain_community/embeddings/openai.py partners/openai/langchain_openai/embeddings/base.py
mv community/langchain_community/embeddings/azure_openai.py partners/openai/langchain_openai/embeddings/azure.py
mv community/langchain_community/adapters/openai.py partners/openai/langchain_openai/adapters.py

mv community/tests/unit_tests/chat_models/test_openai.py partners/openai/tests/unit_tests/chat_models/test_base.py
mv community/tests/integration_tests/chat_models/test_openai.py partners/openai/tests/integration_tests/chat_models/test_base.py
mv community/tests/unit_tests/chat_models/test_azureopenai.py partners/openai/tests/unit_tests/chat_models/test_azure.py
mv community/tests/integration_tests/chat_models/test_azure_openai.py partners/openai/tests/integration_tests/chat_models/test_azure.py

mv community/tests/unit_tests/llms/test_openai.py partners/openai/tests/unit_tests/llms/test_base.py
mv community/tests/integration_tests/llms/test_openai.py partners/openai/tests/integration_tests/llms/test_base.py
mv community/tests/integration_tests/llms/test_azure_openai.py partners/openai/tests/integration_tests/llms/test_azure.py

mv community/tests/unit_tests/embeddings/test_openai.py partners/openai/tests/unit_tests/embeddings/test_base.py
mv community/tests/integration_tests/embeddings/test_openai.py partners/openai/tests/integration_tests/embeddings/test_base.py
mv community/tests/integration_tests/embeddings/test_azure_openai.py partners/openai/tests/integration_tests/embeddings/test_azure.py

mkdir -p partners/openai/tests/integration_tests/adapters
mv community/tests/integration_tests/adapters/{__init__,test_openai}.py partners/openai/tests/integration_tests/adapters

git add partners core

rm community/langchain_community/{chat_models,llms,tools,embeddings,vectorstores,callbacks}/base.py
rm community/tests/unit_tests/{chat_models,llms,tools,callbacks}/test_base.py
rm community/tests/unit_tests/callbacks/test_manager.py
rm community/langchain_community/callbacks/{stdout,streaming_stdout}.py
rm community/langchain_community/callbacks/tracers/{base,evaluation,langchain,langchain_v1,log_stream,root_listeners,run_collector,schemas,stdout}.py

git checkout master -- langchain/tests/unit_tests/{chat_models,llms,tools,callbacks,document_loaders}/test_base.py
git checkout master -- langchain/tests/unit_tests/{callbacks,docstore,document_loaders,document_transformers,embeddings,graphs,llms,chat_models,storage,tools,utilities,vectorstores}/test_imports.py
git checkout master -- langchain/tests/unit_tests/callbacks/test_manager.py
git checkout master -- langchain/tests/unit_tests/document_loaders/blob_loaders/test_public_api.py
git checkout master -- langchain/tests/unit_tests/document_loaders/parsers/test_public_api.py
git checkout master -- langchain/tests/unit_tests/vectorstores/test_public_api.py
git checkout master -- langchain/tests/unit_tests/schema
touch langchain/tests/unit_tests/{llms,chat_models,tools,callbacks,runnables,document_loaders,docstore,document_transformers,embeddings,graphs,storage,utilities,vectorstores}/__init__.py
touch langchain/tests/unit_tests/document_loaders/{blob_loaders,parsers}/__init__.py

cp -r core/scripts community
cp -r core/scripts partners/openai


sed -i '' 's/from\ langchain_openai.chat_models\ /from\ langchain_openai.chat_models.base\ /g' partners/openai/langchain_openai/chat_models/azure.py
sed -i '' 's/from\ langchain_openai.embeddings\ /from\ langchain_openai.embeddings.base\ /g' partners/openai/langchain_openai/embeddings/azure.py

git grep -l '@pytest\.mark\.requires' partners/openai | xargs sed -i '' 's/@pytest\.mark\.requires("openai")//g'


cp -r langchain/tests/integration_tests/examples community/tests
cp -r langchain/tests/integration_tests/examples community/tests/integration_tests
cp -r langchain/tests/unit_tests/examples community/tests/unit_tests
cp langchain/tests/unit_tests/conftest.py community/tests/unit_tests
cp langchain/tests/integration_tests/test_compile.py community/tests/integration_tests
cp langchain/tests/integration_tests/test_compile.py partners/openai/tests/integration_tests
cp -r ../.scripts/community_split/libs/* .
cp community/tests/integration_tests/vectorstores/fake_embeddings.py langchain/tests/integration_tests/cache/

g grep -l 'integration_tests\.vectorstores\.fake_embeddings' community/tests | xargs sed -i '' 's/integration_tests\.vectorstores\.fake_embeddings/integration_tests.cache.fake_embeddings/g'

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
touch community/langchain_community/agent_toolkits/amadeus/__init__.py