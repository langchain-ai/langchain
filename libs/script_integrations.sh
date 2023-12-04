git checkout master -- langchain

rm -rf community/langchain_community
rm -rf partners/openai/langchain_openai
rm -rf community/tests

mkdir -p community/langchain_community
touch community/langchain_community/__init__.py
mkdir -p community/tests
touch community/tests/__init__.py
mkdir -p community/tests/unit_tests
touch community/tests/unit_tests/__init__.py
mkdir -p community/tests/integration_tests/
touch community/tests/integration_tests/__init__.py
mkdir -p community/langchain_community/utils
touch community/langchain_community/utils/__init__.py

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
cp langchain/langchain/utils/math.py community/langchain_community/utils

cd community

git add langchain_community
git grep -l 'from langchain_core.pydantic_v1' | xargs sed -i '' 's/from langchain_core.pydantic_v1/from langchain_core.pydantic_v1/g'
git grep -l 'from langchain_community.utils.math' | xargs sed -i '' 's/from langchain_community.utils.math/from langchain_community.utils.math/g'
git grep -l 'from langchain_openai.functions' | xargs sed -i '' 's/from langchain_openai.functions/from langchain_openai.functions/g'
git grep -l 'from langchain_openai.utils' | xargs sed -i '' 's/from langchain_openai.utils/from langchain_openai.utils/g'
git grep -l 'from langchain_openai.chat_model' | xargs sed -i '' 's/from langchain_openai.chat_model/from langchain_openai.chat_model/g'
git grep -l 'from langchain_openai.embedding' | xargs sed -i '' 's/from langchain_openai.embedding/from langchain_openai.embedding/g'
git grep -l 'from langchain_openai.llm' | xargs sed -i '' 's/from langchain_openai.llm/from langchain_openai.llm/g'

git grep -l 'from langchain_core.tools' | xargs sed -i '' 's/from langchain_core.tools/from langchain_core.tools/g'
git grep -l 'from langchain_core.tools' | xargs sed -i '' 's/from langchain_core.tools/from langchain_core.tools/g'

git grep -l 'from langchain_core.callbacks' | xargs sed -i '' 's/from langchain_core.callbacks/from langchain_core.callbacks/g'
git grep -l 'from langchain_core.outputs' | xargs sed -i '' 's/from langchain_core.outputs/from langchain_core.outputs/g'

git grep -l 'from langchain_core.messages' | xargs sed -i '' 's/from langchain_core.messages/from langchain_core.messages/g'
git grep -l 'from langchain_core.utils' | xargs sed -i '' 's/from langchain_core.utils/from langchain_core.utils/g'
git grep -l 'from langchain\.' | xargs sed -i '' 's/from langchain\./from langchain_community./g'
git grep -l 'from langchain_community.chat_message_histories' | xargs sed -i '' 's/from langchain_community.chat_message_histories/from langchain_community.chat_message_histories/g'
git grep -l 'from langchain_community.agent_toolkits' | xargs sed -i '' 's/from langchain_community.agent_toolkits/from langchain_community.agent_toolkits/g'

cd ..
git checkout master -- langchain
cd langchain
python update_imports.py langchain/chat_loaders langchain_community.chat_loaders
python update_imports.py langchain/document_loaders langchain_community.document_loaders
python update_imports.py langchain/docstore langchain_community.docstore
python update_imports.py langchain/document_transformers langchain_community.document_transformers
python update_imports.py langchain/embeddings langchain_community.embeddings
python update_imports.py langchain/graphs langchain_community.graphs
python update_imports.py langchain/llms langchain_community.llms
python update_imports.py langchain/chat_models langchain_community.chat_models
python update_imports.py langchain/memory/chat_message_histories langchain_community.chat_message_histories
python update_imports.py langchain/storage langchain_community.storage
python update_imports.py langchain/tools langchain_community.tools
python update_imports.py langchain/utilities langchain_community.utilities
python update_imports.py langchain/vectorstores langchain_community.vectorstores
python update_imports.py langchain/adapters langchain_community.adapters
python update_imports.py langchain/agents/agent_toolkits langchain_community.agent_toolkits

#git grep -l 'from langchain_community.llms.base ' | xargs sed -i '' 's/from langchain_community.llms.base /from langchain_core.language_models.llms /g'
#git grep -l 'from langchain_community.chat_models.base ' | xargs sed -i '' 's/from langchain_community.chat_models.base /from langchain_core.language_models.chat_models /g'
#git grep -l 'from langchain_core.tools' | xargs sed -i '' 's/from langchain_core.tools/from langchain_core.tools/g'

git grep -l 'from langchain_openai.llm' | xargs sed -i '' 's/from langchain_openai.llm/from langchain_openai.llm/g'
git grep -l 'from langchain_openai.chat_model' | xargs sed -i '' 's/from langchain_openai.chat_model/from langchain_openai.chat_model/g'
git grep -l 'from langchain_openai.embedding' | xargs sed -i '' 's/from langchain_openai.embedding/from langchain_openai.embedding/g'

sed -i '' '2s/.*/from langchain_community.utilities.requests import TextRequestsWrapper, RequestsWrapper/' langchain/utilities/requests.py
sed -i '' '3s/.*/__all__ = ["Requests", "TextRequestsWrapper", "RequestsWrapper"]/' langchain/utilities/requests.py
sed -i '' '1s/.*/from langchain_community.utilities.asyncio import asyncio_timeout/' langchain/utilities/asyncio.py
sed -i '' '2s/.*/__all__ = ["asyncio_timeout"]/' langchain/utilities/asyncio.py

cd ..
mv community/langchain_community/utilities/loading.py langchain/langchain/utilities
mkdir -p partners/openai/langchain_openai
touch partners/openai/langchain_openai/__init__.py

mv community/langchain_community/chat_models/openai.py partners/openai/langchain_openai/chat_model.py
mv community/langchain_community/llms/openai.py partners/openai/langchain_openai/llm.py
mv community/langchain_community/embeddings/openai.py partners/openai/langchain_openai/embedding.py

cp langchain/langchain/utils/openai.py partners/openai/langchain_openai/utils.py
cp langchain/langchain/utils/openai_functions.py partners/openai/langchain_openai/functions.py

