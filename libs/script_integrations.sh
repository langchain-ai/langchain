git checkout master -- langchain
rm -rf integrations/langchain_integrations
rm -rf partners/openai/langchain_openai
rm -rf integrations/tests
mkdir integrations/langchain_integrations
touch integrations/langchain_integrations/__init__.py
mkdir integrations/tests
touch integrations/tests/__init__.py
mkdir integrations/tests/unit_tests
touch integrations/tests/unit_tests/__init__.py
mkdir integrations/tests/integration_tests/
touch integrations/tests/integration_tests/__init__.py
mv langchain/langchain/chat_loaders integrations/langchain_integrations
mv langchain/langchain/document_loaders integrations/langchain_integrations
mv langchain/langchain/docstore integrations/langchain_integrations
mv langchain/langchain/document_transformers integrations/langchain_integrations
mv langchain/langchain/embeddings integrations/langchain_integrations
mv langchain/langchain/graphs integrations/langchain_integrations
mv langchain/langchain/llms integrations/langchain_integrations
mv langchain/langchain/chat_models integrations/langchain_integrations
mv langchain/langchain/memory/chat_message_histories integrations/langchain_integrations
mv langchain/langchain/storage integrations/langchain_integrations
mv langchain/langchain/tools integrations/langchain_integrations
mv langchain/langchain/utilities integrations/langchain_integrations
mv langchain/langchain/vectorstores integrations/langchain_integrations
mv langchain/langchain/adapters integrations/langchain_integrations
mv langchain/langchain/agents/agent_toolkits integrations/langchain_integrations
mkdir integrations/langchain_integrations/utils
touch integrations/langchain_integrations/utils/__init__.py
cp langchain/langchain/utils/math.py integrations/langchain_integrations/utils
cd integrations
git add langchain_integrations
git grep -l 'from langchain.pydantic_v1' | xargs sed -i '' 's/from langchain.pydantic_v1/from langchain_core.pydantic_v1/g'
git grep -l 'from langchain.utils.math' | xargs sed -i '' 's/from langchain.utils.math/from langchain_integrations.utils.math/g'
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
git grep -l 'from langchain\.' | xargs sed -i '' 's/from langchain\./from langchain_integrations./g'
git grep -l 'from langchain_integrations.memory.chat_message_histories' | xargs sed -i '' 's/from langchain_integrations.memory.chat_message_histories/from langchain_integrations.chat_message_histories/g'
git grep -l 'from langchain_integrations.agents.agent_toolkits' | xargs sed -i '' 's/from langchain_integrations.agents.agent_toolkits/from langchain_integrations.agent_toolkits/g'

cd ..
git checkout master -- langchain
cd langchain
python update_imports.py langchain/chat_loaders langchain_integrations.chat_loaders
python update_imports.py langchain/document_loaders langchain_integrations.document_loaders
python update_imports.py langchain/docstore langchain_integrations.docstore
python update_imports.py langchain/document_transformers langchain_integrations.document_transformers
python update_imports.py langchain/embeddings langchain_integrations.embeddings
python update_imports.py langchain/graphs langchain_integrations.graphs
python update_imports.py langchain/llms langchain_integrations.llms
python update_imports.py langchain/chat_models langchain_integrations.chat_models
python update_imports.py langchain/memory/chat_message_histories langchain_integrations.chat_message_histories
python update_imports.py langchain/storage langchain_integrations.storage
python update_imports.py langchain/tools langchain_integrations.tools
python update_imports.py langchain/utilities langchain_integrations.utilities
python update_imports.py langchain/vectorstores langchain_integrations.vectorstores
python update_imports.py langchain/adapters langchain_integrations.adapters
python update_imports.py langchain/agents/agent_toolkits langchain_integrations.agent_toolkits
git grep -l 'from langchain.llms.base ' | xargs sed -i '' 's/from langchain.llms.base /from langchain_core.language_models.llms /g'
git grep -l 'from langchain.chat_models.base ' | xargs sed -i '' 's/from langchain.chat_models.base /from langchain_core.language_models.chat_models /g'

git grep -l 'from langchain_integrations.llms.openai' | xargs sed -i '' 's/from langchain_integrations.llms.openai/from langchain_openai.llm/g'
git grep -l 'from langchain_integrations.chat_models.openai' | xargs sed -i '' 's/from langchain_integrations.chat_models.openai/from langchain_openai.chat_model/g'
git grep -l 'from langchain_integrations.embeddings.openai' | xargs sed -i '' 's/from langchain_integrations.embeddings.openai/from langchain_openai.embedding/g'
git grep -l 'from langchain.tools.base' | xargs sed -i '' 's/from langchain.tools.base/from langchain_core.tools/g'
sed -i '' '2s/.*/from langchain_integrations.utilities.requests import TextRequestsWrapper, RequestsWrapper/' langchain/utilities/requests.py
sed -i '' '3s/.*/__all__ = ["Requests", "TextRequestsWrapper", "RequestsWrapper"]/' langchain/utilities/requests.py
sed -i '' '1s/.*/from langchain_integrations.utilities.asyncio import asyncio_timeout/' langchain/utilities/asyncio.py
sed -i '' '2s/.*/__all__ = ["asyncio_timeout"]/' langchain/utilities/asyncio.py

cd ..
mv integrations/langchain_integrations/utilities/loading.py langchain/langchain/utilities
mkdir partners/openai/langchain_openai
touch partners/openai/langchain_openai/__init__.py
mv integrations/langchain_integrations/chat_models/openai.py partners/openai/langchain_openai/chat_model.py
mv integrations/langchain_integrations/llms/openai.py partners/openai/langchain_openai/llm.py
mv integrations/langchain_integrations/embeddings/openai.py partners/openai/langchain_openai/embedding.py
cp langchain/langchain/utils/openai.py partners/openai/langchain_openai/utils.py
cp langchain/langchain/utils/openai_functions.py partners/openai/langchain_openai/functions.py



