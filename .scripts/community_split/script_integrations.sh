#!/bin/bash

cd libs

# cleanup anything existing
git checkout master -- langchain/{langchain,tests}
git checkout master -- core/{langchain_core,tests}
git checkout master -- experimental/{langchain_experimental,tests}
rm -rf community/{langchain_community,tests}

# make new dirs
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

# import core stuff from core
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

# mv stuff to community
cd ..

mv langchain/langchain/adapters community/langchain_community
mv langchain/langchain/callbacks community/langchain_community/callbacks
mv langchain/langchain/chat_loaders community/langchain_community
mv langchain/langchain/chat_models community/langchain_community
mv langchain/langchain/document_loaders community/langchain_community
mv langchain/langchain/docstore community/langchain_community
mv langchain/langchain/document_transformers community/langchain_community
mv langchain/langchain/embeddings community/langchain_community
mv langchain/langchain/graphs community/langchain_community
mv langchain/langchain/llms community/langchain_community
mv langchain/langchain/memory/chat_message_histories community/langchain_community
mv langchain/langchain/retrievers community/langchain_community
mv langchain/langchain/storage community/langchain_community
mv langchain/langchain/tools community/langchain_community
mv langchain/langchain/utilities community/langchain_community
mv langchain/langchain/vectorstores community/langchain_community

mv langchain/langchain/agents/agent_toolkits community/langchain_community
mv langchain/langchain/cache.py community/langchain_community
mv langchain/langchain/indexes/base.py community/langchain_community/indexes
mv langchain/langchain/indexes/_sql_record_manager.py community/langchain_community/indexes
mv langchain/langchain/utils/{math,openai,openai_functions}.py community/langchain_community/utils

# mv stuff to core
mv langchain/langchain/utils/json_schema.py core/langchain_core/utils
mv langchain/langchain/utils/html.py core/langchain_core/utils
mv langchain/langchain/utils/strings.py core/langchain_core/utils
cat langchain/langchain/utils/env.py >> core/langchain_core/utils/env.py
rm langchain/langchain/utils/env.py

# mv unit tests to community
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
mv langchain/tests/unit_tests/retrievers community/tests/unit_tests
mv langchain/tests/unit_tests/callbacks community/tests/unit_tests
mv langchain/tests/unit_tests/indexes/test_sql_record_manager.py community/tests/unit_tests/indexes
mv langchain/tests/unit_tests/utils/test_math.py community/tests/unit_tests/utils

# cp some test helpers back to langchain
mkdir -p langchain/tests/unit_tests/llms
cp {community,langchain}/tests/unit_tests/llms/fake_llm.py
cp {community,langchain}/tests/unit_tests/llms/fake_chat_model.py
mkdir -p langchain/tests/unit_tests/callbacks
cp {community,langchain}/tests/unit_tests/callbacks/fake_callback_handler.py

# mv unit tests to core
mv langchain/tests/unit_tests/utils/test_json_schema.py core/tests/unit_tests/utils
mv langchain/tests/unit_tests/utils/test_html.py core/tests/unit_tests/utils

# mv integration tests to community
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
mv langchain/tests/integration_tests/retrievers community/tests/integration_tests
mv langchain/tests/integration_tests/adapters community/tests/integration_tests
mv langchain/tests/integration_tests/callbacks community/tests/integration_tests
mv langchain/tests/integration_tests/{test_kuzu,test_nebulagraph}.py community/tests/integration_tests/graphs
touch community/tests/integration_tests/{chat_message_histories,tools}/__init__.py

# import new core stuff from core (everywhere)
git grep -l 'from langchain.utils.json_schema' | xargs sed -i '' 's/from langchain.utils.json_schema/from langchain_core.utils.json_schema/g'
git grep -l 'from langchain.utils.html' | xargs sed -i '' 's/from langchain.utils.html/from langchain_core.utils.html/g'
git grep -l 'from langchain.utils.strings' | xargs sed -i '' 's/from langchain.utils.strings/from langchain_core.utils.strings/g'
git grep -l 'from langchain.utils.env' | xargs sed -i '' 's/from langchain.utils.env/from langchain_core.utils.env/g'

git add community
cd community

# import core stuff from core
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
git grep -l 'from langchain.schema import BaseRetriever' | xargs sed -i '' 's/from langchain.schema\ import\ BaseRetriever/from langchain_core.retrievers import BaseRetriever/g'
git grep -l 'from langchain.schema import Document' | xargs sed -i '' 's/from langchain.schema\ import\ Document/from langchain_core.documents import Document/g'

# import openai stuff from openai
git grep -l 'from langchain.utils.math' | xargs sed -i '' 's/from langchain.utils.math/from langchain_community.utils.math/g'
git grep -l 'from langchain.utils.openai_functions' | xargs sed -i '' 's/from langchain.utils.openai_functions/from langchain_community.utils.openai_functions/g'
git grep -l 'from langchain.utils.openai' | xargs sed -i '' 's/from langchain.utils.openai/from langchain_community.utils.openai/g'
git grep -l 'from langchain.utils' | xargs sed -i '' 's/from langchain.utils/from langchain_core.utils/g'
git grep -l 'from langchain\.' | xargs sed -i '' 's/from langchain\./from langchain_community./g'
git grep -l 'from langchain_community.memory.chat_message_histories' | xargs sed -i '' 's/from langchain_community.memory.chat_message_histories/from langchain_community.chat_message_histories/g'
git grep -l 'from langchain_community.agents.agent_toolkits' | xargs sed -i '' 's/from langchain_community.agents.agent_toolkits/from langchain_community.agent_toolkits/g'

sed -i '' 's/from\ langchain.chat_models\ import\ ChatOpenAI/from langchain_openai.chat_models import ChatOpenAI/g' langchain_community/chat_models/promptlayer_openai.py

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

# update all moved langchain files to re-export classes and functions
cd ../langchain
git checkout master -- langchain

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
python ../../.scripts/community_split/update_imports.py langchain/retrievers langchain_community.retrievers
python ../../.scripts/community_split/update_imports.py langchain/adapters langchain_community.adapters
python ../../.scripts/community_split/update_imports.py langchain/agents/agent_toolkits langchain_community.agent_toolkits
python ../../.scripts/community_split/update_imports.py langchain/cache.py langchain_community.cache
python ../../.scripts/community_split/update_imports.py langchain/utils/math.py langchain_community.utils.math
python ../../.scripts/community_split/update_imports.py langchain/utils/json_schema.py langchain_core.utils.json_schema
python ../../.scripts/community_split/update_imports.py langchain/utils/html.py langchain_core.utils.html
python ../../.scripts/community_split/update_imports.py langchain/utils/env.py langchain_core.utils.env
python ../../.scripts/community_split/update_imports.py langchain/utils/strings.py langchain_core.utils.strings
python ../../.scripts/community_split/update_imports.py langchain/utils/openai.py langchain_community.utils.openai
python ../../.scripts/community_split/update_imports.py langchain/utils/openai_functions.py langchain_community.utils.openai_functions

# update core and openai imports
git grep -l 'from langchain.llms.base ' | xargs sed -i '' 's/from langchain.llms.base /from langchain_core.language_models.llms /g'
git grep -l 'from langchain.chat_models.base ' | xargs sed -i '' 's/from langchain.chat_models.base /from langchain_core.language_models.chat_models /g'
git grep -l 'from langchain.tools.base' | xargs sed -i '' 's/from langchain.tools.base/from langchain_core.tools/g'

git grep -l 'langchain_core.language_models.llmsten' | xargs sed -i '' 's/langchain_core.language_models.llmsten/langchain_community.llms.baseten/g'

cd ..

mv community/langchain_community/utilities/loading.py langchain/langchain/utilities
mv community/langchain_community/utilities/asyncio.py langchain/langchain/utilities

#git add partners
git add core

# rm files from community that just export core classes
rm community/langchain_community/{chat_models,llms,tools,embeddings,vectorstores,callbacks}/base.py
rm community/tests/unit_tests/{chat_models,llms,tools,callbacks}/test_base.py
rm community/tests/unit_tests/callbacks/test_manager.py
rm community/langchain_community/callbacks/{stdout,streaming_stdout}.py
rm community/langchain_community/callbacks/tracers/{base,evaluation,langchain,langchain_v1,log_stream,root_listeners,run_collector,schemas,stdout}.py
rm community/langchain_community/retrievers/{multi_query,multi_vector,contextual_compression,ensemble,merger_retriever,parent_document_retriever,re_phraser,web_research,time_weighted_retriever}.py
rm -r community/langchain_community/retrievers/{self_query,document_compressors}
rm community/tests/unit_tests/retrievers/test_{ensemble,multi_query,multi_vector,parent_document,time_weighted_retriever,web_research}.py
rm community/tests/integration_tests/retrievers/test_{contextual_compression,merger_retriever}.py
rm -r community/tests/unit_tests/retrievers/{self_query,document_compressors}
rm -r community/tests/integration_tests/retrievers/document_compressors

# keep export tests in langchain
git checkout master -- langchain/tests/unit_tests/{chat_models,llms,tools,callbacks,document_loaders}/test_base.py
git checkout master -- langchain/tests/unit_tests/{callbacks,docstore,document_loaders,document_transformers,embeddings,graphs,llms,chat_models,storage,tools,utilities,vectorstores}/test_imports.py
git checkout master -- langchain/tests/unit_tests/callbacks/test_manager.py
git checkout master -- langchain/tests/unit_tests/document_loaders/blob_loaders/test_public_api.py
git checkout master -- langchain/tests/unit_tests/document_loaders/parsers/test_public_api.py
git checkout master -- langchain/tests/unit_tests/vectorstores/test_public_api.py
git checkout master -- langchain/tests/unit_tests/schema
git checkout master -- langchain/langchain/retrievers/{multi_query,multi_vector,self_query/base,contextual_compression,ensemble,merger_retriever,parent_document_retriever,re_phraser,web_research,time_weighted_retriever}.py
git checkout master -- langchain/langchain/retrievers/{self_query,document_compressors}
git checkout master -- langchain/tests/unit_tests/retrievers/test_{ensemble,multi_query,multi_vector,parent_document,time_weighted_retriever,web_research}.py
git checkout master -- langchain/tests/integration_tests/retrievers/test_{contextual_compression,merger_retriever}.py
git checkout master -- langchain/tests/unit_tests/retrievers/{self_query,document_compressors}
git checkout master -- langchain/tests/integration_tests/retrievers/document_compressors
touch langchain/tests/unit_tests/{llms,chat_models,tools,callbacks,runnables,document_loaders,docstore,document_transformers,embeddings,graphs,storage,utilities,vectorstores,retrievers}/__init__.py
touch langchain/tests/unit_tests/document_loaders/{blob_loaders,parsers}/__init__.py
mv {community,langchain}/tests/unit_tests/retrievers/sequential_retriever.py

# cp lint scripts
cp -r core/scripts community

# cp test helpers
cp -r langchain/tests/integration_tests/examples community/tests
cp -r langchain/tests/integration_tests/examples community/tests/integration_tests
cp -r langchain/tests/unit_tests/examples community/tests/unit_tests
cp langchain/tests/unit_tests/conftest.py community/tests/unit_tests
cp community/tests/integration_tests/vectorstores/fake_embeddings.py langchain/tests/integration_tests/cache/
cp langchain/tests/integration_tests/test_compile.py community/tests/integration_tests

# cp manually changed files
cp -r ../.scripts/community_split/libs/* .

# mv some tests to integrations
mv community/tests/{unit_tests,integration_tests}/document_loaders/test_telegram.py
mv community/tests/{unit_tests,integration_tests}/document_loaders/parsers/test_docai.py
mv community/tests/{unit_tests,integration_tests}/chat_message_histories/test_streamlit.py

# fix some final tests
git grep -l 'integration_tests\.vectorstores\.fake_embeddings' langchain/tests | xargs sed -i '' 's/integration_tests\.vectorstores\.fake_embeddings/integration_tests.cache.fake_embeddings/g'
touch community/langchain_community/agent_toolkits/amadeus/__init__.py

# format
cd core
make format
cd ../langchain
make format
cd ../experimental
make format
cd ../community
make format
g add .

cd ..
sed -E -i '' '1 s/(.*)/\1\ \ \#\ noqa\:\ E501/g' langchain/langchain/agents/agent_toolkits/conversational_retrieval/openai_functions.py
sed -E -i '' 's/import\ importlib$/import importlib.util/g' experimental/langchain_experimental/prompts/load.py
