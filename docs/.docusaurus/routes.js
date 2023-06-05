import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '7d3'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '7b4'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'f92'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', '1b0'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '33d'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '79d'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', 'd7b'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', 'f47'),
    routes: [
      {
        path: '/docs/',
        component: ComponentCreator('/docs/', '890'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/additional_resources/tracing',
        component: ComponentCreator('/docs/additional_resources/tracing', '84a'),
        exact: true
      },
      {
        path: '/docs/additional_resources/youtube',
        component: ComponentCreator('/docs/additional_resources/youtube', '56a'),
        exact: true
      },
      {
        path: '/docs/api/',
        component: ComponentCreator('/docs/api/', '733'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/api/API',
        component: ComponentCreator('/docs/api/API', '6c1'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/dependents',
        component: ComponentCreator('/docs/dependents', 'f7c'),
        exact: true
      },
      {
        path: '/docs/ecosystem/deployments',
        component: ComponentCreator('/docs/ecosystem/deployments', '722'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/ecosystem/modelscope',
        component: ComponentCreator('/docs/ecosystem/modelscope', '22b'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/get_started/quickstart',
        component: ComponentCreator('/docs/get_started/quickstart', '7c1'),
        exact: true
      },
      {
        path: '/docs/getting_started/installation',
        component: ComponentCreator('/docs/getting_started/installation', '714'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/integrations/ai21',
        component: ComponentCreator('/docs/integrations/ai21', 'd0d'),
        exact: true
      },
      {
        path: '/docs/integrations/airbyte',
        component: ComponentCreator('/docs/integrations/airbyte', 'e40'),
        exact: true
      },
      {
        path: '/docs/integrations/aleph_alpha',
        component: ComponentCreator('/docs/integrations/aleph_alpha', '3bb'),
        exact: true
      },
      {
        path: '/docs/integrations/analyticdb',
        component: ComponentCreator('/docs/integrations/analyticdb', 'a5f'),
        exact: true
      },
      {
        path: '/docs/integrations/annoy',
        component: ComponentCreator('/docs/integrations/annoy', '195'),
        exact: true
      },
      {
        path: '/docs/integrations/anyscale',
        component: ComponentCreator('/docs/integrations/anyscale', 'ccf'),
        exact: true
      },
      {
        path: '/docs/integrations/apify',
        component: ComponentCreator('/docs/integrations/apify', 'fb7'),
        exact: true
      },
      {
        path: '/docs/integrations/argilla',
        component: ComponentCreator('/docs/integrations/argilla', 'e6e'),
        exact: true
      },
      {
        path: '/docs/integrations/arxiv',
        component: ComponentCreator('/docs/integrations/arxiv', '45a'),
        exact: true
      },
      {
        path: '/docs/integrations/atlas',
        component: ComponentCreator('/docs/integrations/atlas', '6e6'),
        exact: true
      },
      {
        path: '/docs/integrations/aws_s3',
        component: ComponentCreator('/docs/integrations/aws_s3', '7b1'),
        exact: true
      },
      {
        path: '/docs/integrations/azlyrics',
        component: ComponentCreator('/docs/integrations/azlyrics', 'aa8'),
        exact: true
      },
      {
        path: '/docs/integrations/azure_blob_storage',
        component: ComponentCreator('/docs/integrations/azure_blob_storage', '42d'),
        exact: true
      },
      {
        path: '/docs/integrations/azure_cognitive_search_',
        component: ComponentCreator('/docs/integrations/azure_cognitive_search_', '739'),
        exact: true
      },
      {
        path: '/docs/integrations/azure_openai',
        component: ComponentCreator('/docs/integrations/azure_openai', '8ed'),
        exact: true
      },
      {
        path: '/docs/integrations/bananadev',
        component: ComponentCreator('/docs/integrations/bananadev', '810'),
        exact: true
      },
      {
        path: '/docs/integrations/beam',
        component: ComponentCreator('/docs/integrations/beam', '469'),
        exact: true
      },
      {
        path: '/docs/integrations/bedrock',
        component: ComponentCreator('/docs/integrations/bedrock', 'd5b'),
        exact: true
      },
      {
        path: '/docs/integrations/bilibili',
        component: ComponentCreator('/docs/integrations/bilibili', '957'),
        exact: true
      },
      {
        path: '/docs/integrations/blackboard',
        component: ComponentCreator('/docs/integrations/blackboard', 'ce2'),
        exact: true
      },
      {
        path: '/docs/integrations/cassandra',
        component: ComponentCreator('/docs/integrations/cassandra', '6f2'),
        exact: true
      },
      {
        path: '/docs/integrations/cerebriumai',
        component: ComponentCreator('/docs/integrations/cerebriumai', 'e9b'),
        exact: true
      },
      {
        path: '/docs/integrations/chroma',
        component: ComponentCreator('/docs/integrations/chroma', '827'),
        exact: true
      },
      {
        path: '/docs/integrations/cohere',
        component: ComponentCreator('/docs/integrations/cohere', '889'),
        exact: true
      },
      {
        path: '/docs/integrations/college_confidential',
        component: ComponentCreator('/docs/integrations/college_confidential', '849'),
        exact: true
      },
      {
        path: '/docs/integrations/confluence',
        component: ComponentCreator('/docs/integrations/confluence', '948'),
        exact: true
      },
      {
        path: '/docs/integrations/ctransformers',
        component: ComponentCreator('/docs/integrations/ctransformers', '693'),
        exact: true
      },
      {
        path: '/docs/integrations/databerry',
        component: ComponentCreator('/docs/integrations/databerry', '974'),
        exact: true
      },
      {
        path: '/docs/integrations/deepinfra',
        component: ComponentCreator('/docs/integrations/deepinfra', 'f13'),
        exact: true
      },
      {
        path: '/docs/integrations/deeplake',
        component: ComponentCreator('/docs/integrations/deeplake', 'ee4'),
        exact: true
      },
      {
        path: '/docs/integrations/diffbot',
        component: ComponentCreator('/docs/integrations/diffbot', 'e03'),
        exact: true
      },
      {
        path: '/docs/integrations/discord',
        component: ComponentCreator('/docs/integrations/discord', '783'),
        exact: true
      },
      {
        path: '/docs/integrations/docugami',
        component: ComponentCreator('/docs/integrations/docugami', '6e5'),
        exact: true
      },
      {
        path: '/docs/integrations/duckdb',
        component: ComponentCreator('/docs/integrations/duckdb', '537'),
        exact: true
      },
      {
        path: '/docs/integrations/elasticsearch',
        component: ComponentCreator('/docs/integrations/elasticsearch', '4c0'),
        exact: true
      },
      {
        path: '/docs/integrations/evernote',
        component: ComponentCreator('/docs/integrations/evernote', 'aa5'),
        exact: true
      },
      {
        path: '/docs/integrations/facebook_chat',
        component: ComponentCreator('/docs/integrations/facebook_chat', 'dd4'),
        exact: true
      },
      {
        path: '/docs/integrations/figma',
        component: ComponentCreator('/docs/integrations/figma', '1b0'),
        exact: true
      },
      {
        path: '/docs/integrations/forefrontai',
        component: ComponentCreator('/docs/integrations/forefrontai', 'b87'),
        exact: true
      },
      {
        path: '/docs/integrations/git',
        component: ComponentCreator('/docs/integrations/git', '355'),
        exact: true
      },
      {
        path: '/docs/integrations/gitbook',
        component: ComponentCreator('/docs/integrations/gitbook', '8bb'),
        exact: true
      },
      {
        path: '/docs/integrations/google_bigquery',
        component: ComponentCreator('/docs/integrations/google_bigquery', '9f1'),
        exact: true
      },
      {
        path: '/docs/integrations/google_cloud_storage',
        component: ComponentCreator('/docs/integrations/google_cloud_storage', '9fe'),
        exact: true
      },
      {
        path: '/docs/integrations/google_drive',
        component: ComponentCreator('/docs/integrations/google_drive', '7fe'),
        exact: true
      },
      {
        path: '/docs/integrations/google_search',
        component: ComponentCreator('/docs/integrations/google_search', '961'),
        exact: true
      },
      {
        path: '/docs/integrations/google_serper',
        component: ComponentCreator('/docs/integrations/google_serper', '33c'),
        exact: true
      },
      {
        path: '/docs/integrations/gooseai',
        component: ComponentCreator('/docs/integrations/gooseai', '1e8'),
        exact: true
      },
      {
        path: '/docs/integrations/gpt4all',
        component: ComponentCreator('/docs/integrations/gpt4all', '006'),
        exact: true
      },
      {
        path: '/docs/integrations/graphsignal',
        component: ComponentCreator('/docs/integrations/graphsignal', '77c'),
        exact: true
      },
      {
        path: '/docs/integrations/gutenberg',
        component: ComponentCreator('/docs/integrations/gutenberg', '5ab'),
        exact: true
      },
      {
        path: '/docs/integrations/hacker_news',
        component: ComponentCreator('/docs/integrations/hacker_news', '1cb'),
        exact: true
      },
      {
        path: '/docs/integrations/hazy_research',
        component: ComponentCreator('/docs/integrations/hazy_research', '884'),
        exact: true
      },
      {
        path: '/docs/integrations/helicone',
        component: ComponentCreator('/docs/integrations/helicone', '0ba'),
        exact: true
      },
      {
        path: '/docs/integrations/huggingface',
        component: ComponentCreator('/docs/integrations/huggingface', '0f0'),
        exact: true
      },
      {
        path: '/docs/integrations/ifixit',
        component: ComponentCreator('/docs/integrations/ifixit', '932'),
        exact: true
      },
      {
        path: '/docs/integrations/imsdb',
        component: ComponentCreator('/docs/integrations/imsdb', 'a79'),
        exact: true
      },
      {
        path: '/docs/integrations/jina',
        component: ComponentCreator('/docs/integrations/jina', '012'),
        exact: true
      },
      {
        path: '/docs/integrations/lancedb',
        component: ComponentCreator('/docs/integrations/lancedb', '41f'),
        exact: true
      },
      {
        path: '/docs/integrations/llamacpp',
        component: ComponentCreator('/docs/integrations/llamacpp', 'f0e'),
        exact: true
      },
      {
        path: '/docs/integrations/mediawikidump',
        component: ComponentCreator('/docs/integrations/mediawikidump', '8f9'),
        exact: true
      },
      {
        path: '/docs/integrations/metal',
        component: ComponentCreator('/docs/integrations/metal', '74b'),
        exact: true
      },
      {
        path: '/docs/integrations/microsoft_onedrive',
        component: ComponentCreator('/docs/integrations/microsoft_onedrive', 'feb'),
        exact: true
      },
      {
        path: '/docs/integrations/microsoft_powerpoint',
        component: ComponentCreator('/docs/integrations/microsoft_powerpoint', '6d8'),
        exact: true
      },
      {
        path: '/docs/integrations/microsoft_word',
        component: ComponentCreator('/docs/integrations/microsoft_word', 'fc5'),
        exact: true
      },
      {
        path: '/docs/integrations/milvus',
        component: ComponentCreator('/docs/integrations/milvus', '71e'),
        exact: true
      },
      {
        path: '/docs/integrations/modal',
        component: ComponentCreator('/docs/integrations/modal', 'bbf'),
        exact: true
      },
      {
        path: '/docs/integrations/modern_treasury',
        component: ComponentCreator('/docs/integrations/modern_treasury', '3d5'),
        exact: true
      },
      {
        path: '/docs/integrations/momento',
        component: ComponentCreator('/docs/integrations/momento', 'fc1'),
        exact: true
      },
      {
        path: '/docs/integrations/myscale',
        component: ComponentCreator('/docs/integrations/myscale', '98f'),
        exact: true
      },
      {
        path: '/docs/integrations/nlpcloud',
        component: ComponentCreator('/docs/integrations/nlpcloud', 'e28'),
        exact: true
      },
      {
        path: '/docs/integrations/notion',
        component: ComponentCreator('/docs/integrations/notion', '1e7'),
        exact: true
      },
      {
        path: '/docs/integrations/obsidian',
        component: ComponentCreator('/docs/integrations/obsidian', '43d'),
        exact: true
      },
      {
        path: '/docs/integrations/openai',
        component: ComponentCreator('/docs/integrations/openai', '236'),
        exact: true
      },
      {
        path: '/docs/integrations/opensearch',
        component: ComponentCreator('/docs/integrations/opensearch', 'b15'),
        exact: true
      },
      {
        path: '/docs/integrations/openweathermap',
        component: ComponentCreator('/docs/integrations/openweathermap', 'c4f'),
        exact: true
      },
      {
        path: '/docs/integrations/petals',
        component: ComponentCreator('/docs/integrations/petals', '2d1'),
        exact: true
      },
      {
        path: '/docs/integrations/pgvector',
        component: ComponentCreator('/docs/integrations/pgvector', 'bac'),
        exact: true
      },
      {
        path: '/docs/integrations/pinecone',
        component: ComponentCreator('/docs/integrations/pinecone', 'dfc'),
        exact: true
      },
      {
        path: '/docs/integrations/pipelineai',
        component: ComponentCreator('/docs/integrations/pipelineai', 'ed7'),
        exact: true
      },
      {
        path: '/docs/integrations/predictionguard',
        component: ComponentCreator('/docs/integrations/predictionguard', 'a04'),
        exact: true
      },
      {
        path: '/docs/integrations/promptlayer',
        component: ComponentCreator('/docs/integrations/promptlayer', '04b'),
        exact: true
      },
      {
        path: '/docs/integrations/psychic',
        component: ComponentCreator('/docs/integrations/psychic', '267'),
        exact: true
      },
      {
        path: '/docs/integrations/qdrant',
        component: ComponentCreator('/docs/integrations/qdrant', '606'),
        exact: true
      },
      {
        path: '/docs/integrations/reddit',
        component: ComponentCreator('/docs/integrations/reddit', 'd8b'),
        exact: true
      },
      {
        path: '/docs/integrations/redis',
        component: ComponentCreator('/docs/integrations/redis', 'bcb'),
        exact: true
      },
      {
        path: '/docs/integrations/replicate',
        component: ComponentCreator('/docs/integrations/replicate', '6e3'),
        exact: true
      },
      {
        path: '/docs/integrations/roam',
        component: ComponentCreator('/docs/integrations/roam', '97c'),
        exact: true
      },
      {
        path: '/docs/integrations/runhouse',
        component: ComponentCreator('/docs/integrations/runhouse', '373'),
        exact: true
      },
      {
        path: '/docs/integrations/rwkv',
        component: ComponentCreator('/docs/integrations/rwkv', 'e8a'),
        exact: true
      },
      {
        path: '/docs/integrations/sagemaker_endpoint',
        component: ComponentCreator('/docs/integrations/sagemaker_endpoint', '60f'),
        exact: true
      },
      {
        path: '/docs/integrations/searx',
        component: ComponentCreator('/docs/integrations/searx', '4f1'),
        exact: true
      },
      {
        path: '/docs/integrations/serpapi',
        component: ComponentCreator('/docs/integrations/serpapi', '5b8'),
        exact: true
      },
      {
        path: '/docs/integrations/sklearn',
        component: ComponentCreator('/docs/integrations/sklearn', 'c0c'),
        exact: true
      },
      {
        path: '/docs/integrations/slack',
        component: ComponentCreator('/docs/integrations/slack', '0d4'),
        exact: true
      },
      {
        path: '/docs/integrations/spacy',
        component: ComponentCreator('/docs/integrations/spacy', '416'),
        exact: true
      },
      {
        path: '/docs/integrations/spreedly',
        component: ComponentCreator('/docs/integrations/spreedly', 'fd2'),
        exact: true
      },
      {
        path: '/docs/integrations/stochasticai',
        component: ComponentCreator('/docs/integrations/stochasticai', '5bc'),
        exact: true
      },
      {
        path: '/docs/integrations/stripe',
        component: ComponentCreator('/docs/integrations/stripe', '7d0'),
        exact: true
      },
      {
        path: '/docs/integrations/tair',
        component: ComponentCreator('/docs/integrations/tair', '69c'),
        exact: true
      },
      {
        path: '/docs/integrations/telegram',
        component: ComponentCreator('/docs/integrations/telegram', '31b'),
        exact: true
      },
      {
        path: '/docs/integrations/tomarkdown',
        component: ComponentCreator('/docs/integrations/tomarkdown', 'f42'),
        exact: true
      },
      {
        path: '/docs/integrations/trello',
        component: ComponentCreator('/docs/integrations/trello', '532'),
        exact: true
      },
      {
        path: '/docs/integrations/twitter',
        component: ComponentCreator('/docs/integrations/twitter', 'a8d'),
        exact: true
      },
      {
        path: '/docs/integrations/unstructured',
        component: ComponentCreator('/docs/integrations/unstructured', '311'),
        exact: true
      },
      {
        path: '/docs/integrations/vectara',
        component: ComponentCreator('/docs/integrations/vectara', '151'),
        exact: true
      },
      {
        path: '/docs/integrations/vespa',
        component: ComponentCreator('/docs/integrations/vespa', '033'),
        exact: true
      },
      {
        path: '/docs/integrations/weather',
        component: ComponentCreator('/docs/integrations/weather', 'b38'),
        exact: true
      },
      {
        path: '/docs/integrations/weaviate',
        component: ComponentCreator('/docs/integrations/weaviate', '31d'),
        exact: true
      },
      {
        path: '/docs/integrations/whatsapp',
        component: ComponentCreator('/docs/integrations/whatsapp', '6dd'),
        exact: true
      },
      {
        path: '/docs/integrations/wikipedia',
        component: ComponentCreator('/docs/integrations/wikipedia', 'ce8'),
        exact: true
      },
      {
        path: '/docs/integrations/wolfram_alpha',
        component: ComponentCreator('/docs/integrations/wolfram_alpha', 'b76'),
        exact: true
      },
      {
        path: '/docs/integrations/writer',
        component: ComponentCreator('/docs/integrations/writer', '435'),
        exact: true
      },
      {
        path: '/docs/integrations/yeagerai',
        component: ComponentCreator('/docs/integrations/yeagerai', '9f2'),
        exact: true
      },
      {
        path: '/docs/integrations/youtube',
        component: ComponentCreator('/docs/integrations/youtube', '415'),
        exact: true
      },
      {
        path: '/docs/integrations/zep',
        component: ComponentCreator('/docs/integrations/zep', '965'),
        exact: true
      },
      {
        path: '/docs/integrations/zilliz',
        component: ComponentCreator('/docs/integrations/zilliz', '42e'),
        exact: true
      },
      {
        path: '/docs/modules/agents/agents/agent_types',
        component: ComponentCreator('/docs/modules/agents/agents/agent_types', 'eb3'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/modules/agents/tools/getting_started',
        component: ComponentCreator('/docs/modules/agents/tools/getting_started', 'ac3'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/modules/data_io',
        component: ComponentCreator('/docs/modules/data_io', '651'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/modules/model_io',
        component: ComponentCreator('/docs/modules/model_io', '2f9'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/modules/model_io/prompts/example_selectors/examples/custom_example_selector',
        component: ComponentCreator('/docs/modules/model_io/prompts/example_selectors/examples/custom_example_selector', 'be6'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/modules/model_io/prompts/prompt_templates/getting_started',
        component: ComponentCreator('/docs/modules/model_io/prompts/prompt_templates/getting_started', '66b'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/templates/integration',
        component: ComponentCreator('/docs/templates/integration', '601'),
        exact: true
      },
      {
        path: '/docs/tracing/hosted_installation',
        component: ComponentCreator('/docs/tracing/hosted_installation', 'af8'),
        exact: true
      },
      {
        path: '/docs/tracing/local_installation',
        component: ComponentCreator('/docs/tracing/local_installation', '1bc'),
        exact: true
      },
      {
        path: '/docs/use_cases/agent_simulations',
        component: ComponentCreator('/docs/use_cases/agent_simulations', '01c'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/use_cases/apis',
        component: ComponentCreator('/docs/use_cases/apis', '559'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/use_cases/autonomous_agents',
        component: ComponentCreator('/docs/use_cases/autonomous_agents', '9c6'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/use_cases/chatbots',
        component: ComponentCreator('/docs/use_cases/chatbots', '275'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/use_cases/code',
        component: ComponentCreator('/docs/use_cases/code', 'bc9'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/use_cases/extraction',
        component: ComponentCreator('/docs/use_cases/extraction', '447'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/use_cases/personal_assistants',
        component: ComponentCreator('/docs/use_cases/personal_assistants', '639'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/use_cases/question_answering',
        component: ComponentCreator('/docs/use_cases/question_answering', '083'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/use_cases/summarization',
        component: ComponentCreator('/docs/use_cases/summarization', '1cb'),
        exact: true,
        sidebar: "sidebar"
      },
      {
        path: '/docs/use_cases/tabular',
        component: ComponentCreator('/docs/use_cases/tabular', '9ed'),
        exact: true,
        sidebar: "sidebar"
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '534'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
