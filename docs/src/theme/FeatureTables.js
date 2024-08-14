import React from "react";
import {useCurrentSidebarCategory} from '@docusaurus/theme-common';
import {
  useDocById,
} from '@docusaurus/theme-common/internal';

const FEATURE_TABLES = {
    chat: {
        link: "/docs/integrations/chat",
        columns: [
            {title: "Provider", formatter: (item) => <a href={item.link}>{item.name}</a>},
            {title: <a href="/docs/how_to/tool_calling">Tool calling</a>, formatter: (item) => item.tool_calling ? "✅" : "❌"},
            {title: <a href="/docs/how_to/structured_output/">Structured output</a>, formatter: (item) => item.structured_output ? "✅" : "❌"},
            {title: "JSON mode", formatter: (item) => item.json_mode ? "✅" : "❌"},
            {title: "Local", formatter: (item) => item.local ? "✅" : "❌"},
            {title: <a href="/docs/how_to/multimodal_inputs/">Multimodal</a>, formatter: (item) => item.multimodal ? "✅" : "❌"},
            {title: "Package", formatter: (item) => <a href={item.apiLink}>{item.package}</a>},
        ],
        items: [
            {
                "name": "ChatAnthropic",
                "package": "langchain-anthropic",
                "link": "anthropic/",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": false,
                "multimodal": true,
                "local": false,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html#langchain_anthropic.chat_models.ChatAnthropic"
            },
            {
                "name": "ChatMistralAI",
                "package": "langchain-mistralai",
                "link": "mistralai/",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": false,
                "multimodal": false,
                "local": false,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_mistralai.chat_models.ChatMistralAI.html#langchain_mistralai.chat_models.ChatMistralAI"
            },
            {
                "name": "ChatFireworks",
                "package": "langchain-fireworks",
                "link": "fireworks/",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": true,
                "multimodal": false,
                "local": false,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_fireworks.chat_models.ChatFireworks.html#langchain_fireworks.chat_models.ChatFireworks"
            },
            {
                "name": "AzureChatOpenAI",
                "package": "langchain-openai",
                "link": "azure_chat_openai/",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": true,
                "multimodal": true,
                "local": false,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.azure.AzureChatOpenAI.html#langchain_openai.chat_models.azure.AzureChatOpenAI"
            },
            {
                "name": "ChatOpenAI",
                "package": "langchain-openai",
                "link": "openai/",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": true,
                "multimodal": true,
                "local": false,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html#langchain_openai.chat_models.base.ChatOpenAI"
            },
            {
                "name": "ChatTogether",
                "package": "langchain-together",
                "link": "together/",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": true,
                "multimodal": false,
                "local": false,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_together.chat_models.ChatTogether.html#langchain_together.chat_models.ChatTogether"
            },
            {
                "name": "ChatVertexAI",
                "package": "langchain-google-vertexai",
                "link": "google_vertex_ai_palm/",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": false,
                "multimodal": true,
                "local": false,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_google_vertexai.chat_models.ChatVertexAI.html#langchain_google_vertexai.chat_models.ChatVertexAI"
            },
            {
                "name": "ChatGoogleGenerativeAI",
                "package": "langchain-google-genai",
                "link": "google_generative_ai/",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": false,
                "multimodal": true,
                "local": false,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html#langchain_google_genai.chat_models.ChatGoogleGenerativeAI"
            },
            {
                "name": "ChatGroq",
                "package": "langchain-groq",
                "link": "groq/",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": true,
                "multimodal": false,
                "local": false,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_groq.chat_models.ChatGroq.html#langchain_groq.chat_models.ChatGroq"
            },
            {
                "name": "ChatCohere",
                "package": "langchain-cohere",
                "link": "cohere/",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": false,
                "multimodal": false,
                "local": false,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_cohere.chat_models.ChatCohere.html#langchain_cohere.chat_models.ChatCohere"
            },
            {
                "name": "ChatBedrock",
                "package": "langchain-aws",
                "link": "bedrock/",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": false,
                "multimodal": false,
                "local": false,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_aws.chat_models.bedrock.ChatBedrock.html#langchain_aws.chat_models.bedrock.ChatBedrock"
            },
            {
                "name": "ChatHuggingFace",
                "package": "langchain-huggingface",
                "link": "huggingface/",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": false,
                "multimodal": false,
                "local": true,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html#langchain_huggingface.chat_models.huggingface.ChatHuggingFace",
            },
            {
                "name": "ChatNVIDIA",
                "package": "langchain-nvidia-ai-endpoints",
                "link": "nvidia_ai_endpoints/",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": false,
                "multimodal": false,
                "local": true,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_nvidia_ai_endpoints.chat_models.ChatNVIDIA.html#langchain_nvidia_ai_endpoints.chat_models.ChatNVIDIA"
            },
            {
                "name": "ChatOllama",
                "package": "langchain-ollama",
                "link": "ollama/",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": true,
                "multimodal": false,
                "local": true,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_ollama.chat_models.ChatOllama.html#langchain_ollama.chat_models.ChatOllama"
            },
            {
                "name": "ChatLlamaCpp",
                "package": "langchain-community",
                "link": "llamacpp",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": false,
                "multimodal": false,
                "local": true,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.llamacpp.ChatLlamaCpp.html#langchain_community.chat_models.llamacpp.ChatLlamaCpp"
            },
            {
                "name": "ChatAI21",
                "package": "langchain-ai21",
                "link": "ai21",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": false,
                "multimodal": false,
                "local": false,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_ai21.chat_models.ChatAI21.html#langchain_ai21.chat_models.ChatAI21"
            },
            {
                "name": "ChatUpstage",
                "package": "langchain-upstage",
                "link": "upstage",
                "structured_output": true,
                "tool_calling": true,
                "json_mode": false, 
                "multimodal": false,
                "local": false,
                "apiLink": "https://api.python.langchain.com/en/latest/chat_models/langchain_upstage.chat_models.ChatUpstage.html#langchain_upstage.chat_models.ChatUpstage"
            }
        ],
    },
    llms: {
        link: "/docs/integrations/llms",
        columns: [
            {title: "Provider", formatter: (item) => <a href={
                item.link
            }>{item.name}</a>},
            {title: "Package", formatter: (item) => <a href={
                item.apiLink
            }>{item.package}</a>},
        ],
        items: [
            {
                name: "AI21LLM",
                link: "ai21",
                package: "langchain-ai21",
                apiLink: "https://api.python.langchain.com/en/latest/llms/langchain_ai21.llms.AI21LLM.html#langchain_ai21.llms.AI21LLM"
            },
            {
                name: "AnthropicLLM",
                link: "anthropic",
                package: "langchain-anthropic",
                apiLink: "https://api.python.langchain.com/en/latest/llms/langchain_anthropic.llms.AnthropicLLM.html#langchain_anthropic.llms.AnthropicLLM"
            },
            {
                name: "AzureOpenAI",
                link: "azure_openai",
                package: "langchain-openai",
                apiLink: "https://api.python.langchain.com/en/latest/llms/langchain_openai.llms.azure.AzureOpenAI.html#langchain_openai.llms.azure.AzureOpenAI"
            },
            {
                name: "BedrockLLM",
                link: "bedrock",
                package: "langchain-aws",
                apiLink: "https://api.python.langchain.com/en/latest/llms/langchain_aws.llms.bedrock.BedrockLLM.html#langchain_aws.llms.bedrock.BedrockLLM"
            },
            {
                name: "CohereLLM",
                link: "cohere",
                package: "langchain-cohere",
                apiLink: "https://api.python.langchain.com/en/latest/llms/langchain_cohere.llms.Cohere.html#langchain_cohere.llms.Cohere"
            },
            {
                name: "FireworksLLM",
                link: "fireworks",
                package: "langchain-fireworks",
                apiLink: "https://api.python.langchain.com/en/latest/llms/langchain_fireworks.llms.Fireworks.html#langchain_fireworks.llms.Fireworks"
            },
            {
                name: "OllamaLLM",
                link: "ollama",
                package: "langchain-ollama",
                apiLink: "https://api.python.langchain.com/en/latest/llms/langchain_ollama.llms.OllamaLLM.html#langchain_ollama.llms.OllamaLLM"
            },
            {
                name: "OpenAILLM",
                link: "openai",
                package: "langchain-openai",
                apiLink: "https://api.python.langchain.com/en/latest/llms/langchain_openai.llms.base.OpenAI.html#langchain_openai.llms.base.OpenAI"
            },
            {
                name: "TogetherLLM",
                link: "together",
                package: "langchain-together",
                apiLink: "https://api.python.langchain.com/en/latest/llms/langchain_together.llms.Together.html#langchain_together.llms.Together"
            },
            {
                name: "VertexAILLM",
                link: "google_vertexai",
                package: "langchain-google_vertexai",
                apiLink: "https://api.python.langchain.com/en/latest/llms/langchain_google_vertexai.llms.VertexAI.html#langchain_google_vertexai.llms.VertexAI"
            },
        ],
    },
    text_embedding: {
        link: "/docs/integrations/text_embedding",
        columns: [
            {title: "Provider", formatter: (item) => <a href={item.link}>{item.name}</a>},
            {title: "Package", formatter: (item) => <a href={item.apiLink}>{item.package}</a>},
        ],
        items:[
            {
                name: "AzureOpenAI",
                link: "azureopenai",
                package: "langchain-openai",
                apiLink: "https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.azure.AzureOpenAIEmbeddings.html#langchain_openai.embeddings.azure.AzureOpenAIEmbeddings"
            },
            {
                name: "Ollama",
                link: "ollama",
                package: "langchain-ollama",
                apiLink: "https://api.python.langchain.com/en/latest/embeddings/langchain_ollama.embeddings.OllamaEmbeddings.html#langchain_ollama.embeddings.OllamaEmbeddings"
            },
            {
                name: "AI21",
                link: "ai21",
                package: "langchain-ai21",
                apiLink: "https://api.python.langchain.com/en/latest/embeddings/langchain_ai21.embeddings.AI21Embeddings.html#langchain_ai21.embeddings.AI21Embeddings"
            },
            {
                name: "Fake",
                link: "fake",
                package: "langchain-core",
                apiLink: "https://api.python.langchain.com/en/latest/embeddings/langchain_core.embeddings.fake.FakeEmbeddings.html#langchain_core.embeddings.fake.FakeEmbeddings"
            },
            {
                name: "OpenAI",
                link: "openai",
                package: "langchain-openai",
                apiLink: "https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html#langchain_openai.chat_models.base.ChatOpenAI"
            },
            {
                name: "Together",
                link: "together",
                package: "langchain-together",
                apiLink: "https://api.python.langchain.com/en/latest/embeddings/langchain_together.embeddings.TogetherEmbeddings.html#langchain_together.embeddings.TogetherEmbeddings"
            },
            {
                name: "Fireworks",
                link: "fireworks",
                package: "langchain-fireworks",
                apiLink: "https://api.python.langchain.com/en/latest/embeddings/langchain_fireworks.embeddings.FireworksEmbeddings.html#langchain_fireworks.embeddings.FireworksEmbeddings"
            },
            {
                name: "MistralAI",
                link: "mistralai",
                package: "langchain-mistralai",
                apiLink: "https://api.python.langchain.com/en/latest/embeddings/langchain_mistralai.embeddings.MistralAIEmbeddings.html#langchain_mistralai.embeddings.MistralAIEmbeddings"
            },
            {
                name: "Cohere",
                link: "cohere",
                package: "langchain-cohere",
                apiLink: "https://api.python.langchain.com/en/latest/embeddings/langchain_cohere.embeddings.CohereEmbeddings.html#langchain_cohere.embeddings.CohereEmbeddings"
            },
            {
                name: "Nomic",
                link: "cohere",
                package: "langchain-nomic",
                apiLink: "https://api.python.langchain.com/en/latest/embeddings/langchain_nomic.embeddings.NomicEmbeddings.html#langchain_nomic.embeddings.NomicEmbeddings"
            },
        ]
    },
    document_retrievers: {
        link: 'docs/integrations/retrievers',
        columns: [
            {title: "Retriever", formatter: (item) => <a href={item.link}>{item.name}</a>},
            {title: "Self-host", formatter: (item) => item.selfHost ? "✅" : "❌"},
            {title: "Cloud offering", formatter: (item) => item.cloudOffering ? "✅" : "❌"},
            {title: "Package", formatter: (item) => <a href={item.apiLink}>{item.package}</a>},
        ],
        items: [
            {
                name: "AmazonKnowledgeBasesRetriever",
                link: "bedrock",
                selfHost: false,
                cloudOffering: true,
                apiLink: "https://api.python.langchain.com/en/latest/retrievers/langchain_aws.retrievers.bedrock.AmazonKnowledgeBasesRetriever.html",
                package: "langchain_aws"
            },
            {
                name: "AzureAISearchRetriever",
                link: "azure_ai_search",
                selfHost: false,
                cloudOffering: true,
                apiLink: "https://api.python.langchain.com/en/latest/retrievers/langchain_community.retrievers.azure_ai_search.AzureAISearchRetriever.html",
                package: "langchain_community"
            },
            {
                name: "ElasticsearchRetriever",
                link: "elasticsearch_retriever",
                selfHost: true,
                cloudOffering: true,
                apiLink: "https://api.python.langchain.com/en/latest/retrievers/langchain_elasticsearch.retrievers.ElasticsearchRetriever.html",
                package: "langchain_elasticsearch"
            },
            {
                name: "MilvusCollectionHybridSearchRetriever",
                link: "milvus_hybrid_search",
                selfHost: true,
                cloudOffering: false,
                apiLink: "https://api.python.langchain.com/en/latest/retrievers/langchain_milvus.retrievers.milvus_hybrid_search.MilvusCollectionHybridSearchRetriever.html",
                package: "langchain_milvus"
            },
            {
                name: "VertexAISearchRetriever",
                link: "google_vertex_ai_search",
                selfHost: false,
                cloudOffering: true,
                apiLink: "https://api.python.langchain.com/en/latest/vertex_ai_search/langchain_google_community.vertex_ai_search.VertexAISearchRetriever.html",
                package: "langchain_google_community"
            }
        ],
    },
    external_retrievers: {
        link: 'docs/integrations/retrievers',
        columns: [
            {title: "Retriever", formatter: (item) => <a href={
                item.link
            }>{item.name}</a>},
            {title: "Source", formatter: (item) => item.source},
            {title: "Package", formatter: (item) => <a href={
                item.apiLink
            }>{item.package}</a>},
        ],
        items: [
            {
                name: "ArxivRetriever",
                link: "arxiv",
                source: (<>Scholarly articles on <a href="https://arxiv.org/">arxiv.org</a></>),
                apiLink: "https://api.python.langchain.com/en/latest/retrievers/langchain_community.retrievers.arxiv.ArxivRetriever.html",
                package: "langchain_community"
            },
            {
                name: "TavilySearchAPIRetriever",
                link: "tavily",
                source: "Internet search",
                apiLink: "https://api.python.langchain.com/en/latest/retrievers/langchain_community.retrievers.tavily_search_api.TavilySearchAPIRetriever.html",
                package: "langchain_community"
            },
            {
                name: "WikipediaRetriever",
                link: "wikipedia",
                source: (<><a href="https://www.wikipedia.org/">Wikipedia</a> articles</>),
                apiLink: "https://api.python.langchain.com/en/latest/retrievers/langchain_community.retrievers.wikipedia.WikipediaRetriever.html",
                package: "langchain_community"
            }
        ]

    },
    document_loaders: {
        link: 'docs/integrations/loaders',
        columns: [],
        items: [],
    },
    webpage_loaders: {
        link: 'docs/integrations/loaders',
        columns: [
            {title: "Document Loader", formatter: (item) => <a href={
                item.link
            }>{item.name}</a>},
            {title: "Description", formatter: (item) => item.source},
            {title: "Package/API", formatter: (item) => item.api},
        ],
        items: [
            {
                name: "Web",
                link: "web_base",
                source: "Uses urllib and BeautifulSoup to load and parse HTML web pages",
                api: "Package",
                apiLink: "https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html"
            },
            {
                name: "RecursiveURL",
                link: "recursive_url",
                source: "Recursively scrapes all child links from a root URL",
                api: "Package",
                apiLink: "https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.recursive_url_loader.RecursiveUrlLoader.html"
            },
            {
                name: "Sitemap",
                link: "sitemap",
                source: "Scrapes all pages on a given sitemap",
                api: "Package",
                apiLink: "https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.sitemap.SitemapLoader.html"
            },
            {
                name: "Firecrawl",
                link: "firecrawl",
                source: "API service that can be deployed locally, hosted version has free credits.",
                api: "API",
                apiLink: "https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.firecrawl.FireCrawlLoader.html"
            }
        ]
    },
    pdf_loaders: {
        link: 'docs/integrations/loaders',
        columns: [
            {title: "Document Loader", formatter: (item) => <a href={
                item.link
            }>{item.name}</a>},
            {title: "Description", formatter: (item) => item.source},
            {title: "Package/API", formatter: (item) => item.api},
        ],
        items: [
            {
                name: "PyPDF",
                link: "pypdfloader",
                source: "Uses `pypdf` to load and parse PDFs",
                api: "Package",
                apiLink: "https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.pdf.PyPDFLoader.html"
            },
            {
                name: "Unstructured",
                link: "unstructured_file",
                source: "Uses Unstructured's open source library to load PDFs",
                api: "Package",
                apiLink: "https://api.python.langchain.com/en/latest/document_loaders/langchain_unstructured.document_loaders.UnstructuredLoader.html"
            },
            {
                name: "Amazon Textract",
                link: "amazon_textract",
                source: "Uses AWS API to load PDFs",
                api: "API",
                apiLink: "https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.pdf.AmazonTextractPDFLoader.html"
            }
        ]
    },
    common_loaders: {
        link: 'docs/integrations/loaders',
        columns: [
            {title: "Document Loader", formatter: (item) => <a href={
                item.link
            }>{item.name}</a>},
            {title: "Data Type", formatter: (item) => item.source},
        ],
        items: [
            {
                name: "CSVLoader",
                link: "csv",
                source: "CSV files",
                apiLink: "https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.csv_loader.CSVLoader.html"
            },
            {
                name: "DirectoryLoader",
                link: "document_loader_directory",
                source: "All files in a given directory",
                apiLink: "https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.directory.DirectoryLoader.html"
            },
            {
                name: "Unstructured",
                link: "unstructured_file",
                source: "All file types",
                apiLink: "https://api.python.langchain.com/en/latest/document_loaders/langchain_unstructured.document_loaders.UnstructuredLoader.html"
            },
        ]
    },
    vectorstores: {
        link: 'docs/integrations/vectorstores',
        columns: [
            {title: "Vectorstore", formatter: (item) => <a href={
                item.link
            }>{item.name}</a>},
            {title: "Delete by ID", formatter: (item) => item.deleteById ? "✅" : "❌"},
            {title: "Filtering", formatter: (item) => item.filtering ? "✅" : "❌"},
            {title: "Search by Vector", formatter: (item) => item.searchByVector ? "✅" : "❌"},
            {title: "Search with score", formatter: (item) => item.searchWithScore ? "✅" : "❌"},
            {title: "Async", formatter: (item) => item.async ? "✅" : "❌"},
            {title: "Passes Standard Tests", formatter: (item) => item.passesStandardTests ? "✅" : "❌"},
            {title: "Multi Tenancy", formatter: (item) => item.multiTenancy ? "✅" : "❌"},
            {title: "IDs in add Documents", formatter: (item) => item.idsInAddDocuments ? "✅" : "❌"},
            {title: "Local/Cloud", formatter: (item) => item.local ? "Local" : "Cloud"},
        ],
        items: [
            {
                name: "AstraDBVectorStore",
                link: "astradb",
                deleteById: true,
                filtering: true,
                searchByVector: true,
                searchWithScore: true,
                async: true,
                passesStandardTests: false,
                multiTenancy: false,
                local: true,
                idsInAddDocuments: false,
            },
            {
                name: "Chroma",
                link: "chroma",
                deleteById: true,
                filtering: true,
                searchByVector: true,
                searchWithScore: true,
                async: true,
                passesStandardTests: false,
                multiTenancy: false,
                local: true,
                idsInAddDocuments: false,
            },
            {
                name: "Clickhouse",
                link: "clickhouse",
                deleteById: true,
                filtering: true,
                searchByVector: false,
                searchWithScore: true,
                async: false,
                passesStandardTests: false,
                multiTenancy: false,
                local: true,
                idsInAddDocuments: false,
            },
            {
                name: "CouchbaseVectorStore",
                link: "couchbase",
                deleteById: true,
                filtering: true,
                searchByVector: false,
                searchWithScore: true,
                async: true,
                passesStandardTests: false,
                multiTenancy: false,
                local: true,
                idsInAddDocuments: false,
            },
            {
                name: "ElasticsearchStore",
                link: "elasticsearch",
                deleteById: true,
                filtering: true,
                searchByVector: true,
                searchWithScore: false,
                async: true,
                passesStandardTests: false,
                multiTenancy: false,
                local: true,
                idsInAddDocuments: false,
            },
            {
                name: "FAISS",
                link: "faiss",
                deleteById: true,
                filtering: true,
                searchByVector: true,
                searchWithScore: true,
                async: true,
                passesStandardTests: false,
                multiTenancy: false,
                local: true,
                idsInAddDocuments: false,
            },
            {
                name: "InMemoryVectorStore",
                link: "in_memory",
                deleteById: true,
                filtering: true,
                searchByVector: false,
                searchWithScore: true,
                async: true,
                passesStandardTests: false,
                multiTenancy: false,
                local: true,
                idsInAddDocuments: false,
            },
            {
                name: "Milvus",
                link: "milvus",
                deleteById: true,
                filtering: true,
                searchByVector: false,
                searchWithScore: true,
                async: true,
                passesStandardTests: false,
                multiTenancy: false,
                local: true,
                idsInAddDocuments: false,
            },
            {
                name: "MongoDBAtlasVectorSearch",
                link: "mongodb_atlas",
                deleteById: true,
                filtering: true,
                searchByVector: false,
                searchWithScore: false,
                async: true,
                passesStandardTests: false,
                multiTenancy: false,
                local: true,
                idsInAddDocuments: false,
            },
            {
                name: "PGVector",
                link: "pg_vector",
                deleteById: true,
                filtering: true,
                searchByVector: true,
                searchWithScore: true,
                async: true,
                passesStandardTests: false,
                multiTenancy: false,
                local: true,
                idsInAddDocuments: false,
            },
            {
                name: "PineconeVectorStore",
                link: "pinecone",
                deleteById: true,
                filtering: true,
                searchByVector: true,
                searchWithScore: false,
                async: true,
                passesStandardTests: false,
                multiTenancy: false,
                local: true,
                idsInAddDocuments: false,
            },
            {
                name: "QdrantVectorStore",
                link: "qdrant",
                deleteById: true,
                filtering: true,
                searchByVector: true,
                searchWithScore: true,
                async: true,
                passesStandardTests: false,
                multiTenancy: false,
                local: true,
                idsInAddDocuments: false,
            },
            {
                name: "Redis",
                link: "redis",
                deleteById: true,
                filtering: true,
                searchByVector: true,
                searchWithScore: true,
                async: true,
                passesStandardTests: false,
                multiTenancy: false,
                local: true,
                idsInAddDocuments: false,
            }
        ],
    }
};

function toTable(columns, items) {
    const headers = columns.map((col) => col.title);
    return (
        <table>
            <thead>
                <tr>
                    {headers.map((header, i) => <th key={`header-${i}`}>{header}</th>)}
                </tr>
            </thead>
            <tbody>
                {items.map((item, i) => (
                    <tr key={`row-${i}`}>
                        {columns.map((col, j) => <td key={`cell-${i}-${j}`}>{col.formatter(item)}</td>)}
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

export function CategoryTable({ category }) {
    const cat = FEATURE_TABLES[category];
    const rtn = toTable(cat.columns, cat.items);
    return rtn;
}

export function ItemTable({ category, item }) {
    const cat = FEATURE_TABLES[category];
    const row = cat.items.find((i) => i.name === item);
    if (!row) {
        throw new Error(`Item ${item} not found in category ${category}`);
    }
    const rtn = toTable(cat.columns, [row]);
    return rtn;
}

function truncate(str, n) {
    return (str.length > n) ? str.substring(0, n-1) + '...' : str;
}

export function IndexTable() {
    const { items } = useCurrentSidebarCategory();

    const rows = items.filter(item => !item.docId?.endsWith?.('/index')).map(item => ({
        ...item,
        description: useDocById(item.docId ?? undefined)?.description,
    }));
    const rtn = toTable(
        [
            { title: "Label", formatter: (item) => <a href={item.href}>{item.label}</a> },
            { title: "Description", formatter: (item) => truncate(item.description ?? "", 70) },
        ],
        rows,
    );
    return rtn;
}
