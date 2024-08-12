import React from "react";
import {useCurrentSidebarCategory} from '@docusaurus/theme-common';
import {
  useDocById,
} from '@docusaurus/theme-common/internal';

interface Column {
    title: string | React.ReactNode;
    formatter: (item: any) => React.ReactNode;
}
interface Category {
    link: string;
    columns: Column[];
    items: any[];
}

const FeatureTables: Record<string, Category> = {
    llms: {
        link: "/docs/integrations/llms",
        columns: [
            {title: "Provider", formatter: (item) => <a href={item.link}>{item.name}</a>},
            {title: "Package", formatter: (item) => <a href={`https://pypi.org/project/${item.package}/`}>{item.package}</a>},
        ],
        items:[
            {
                name: "Anthropic",
                link: "anthropic.ipynb",
                package: "langchain-anthropic",
            }
        ]
    },
    text_embedding: {
        link: "/docs/integrations/text_embedding",
        columns: [
            {title: "Provider", formatter: (item) => <a href={item.link}>{item.name}</a>},
            {title: "Package", formatter: (item) => <a href={`https://pypi.org/project/${item.package}/`}>{item.package}</a>},
        ],
        items:[
            {
                name: "Cohere",
                link: "cohere.ipynb",
                package: "langchain-cohere",
            }
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

    }
};

function toTable(columns: Column[], items: any[]) {
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

export function CategoryTable({category}: {category: string}) {
    const cat = FeatureTables[category];
    return toTable(cat.columns, cat.items);
}

export function ItemTable({category, item}: {category: string, item: string}) {
    const cat = FeatureTables[category];
    const row = cat.items.find((i) => i.name === item);
    if (!row) {
        throw new Error(`Item ${item} not found in category ${category}`);
    }
    return toTable(cat.columns, [row]);
}

type Row = {
    className: string;
    docId: string;
    href: string;
    label: string;
    type: string;
}

function truncate(str: string, n: number) {
    return (str.length > n) ? str.substring(0, n-1) + '...' : str;
}

export function IndexTable() {
    // docusaurus ts typing is confused, so we cast as Row[]
    const {items} = useCurrentSidebarCategory() as {items: Row[]};
    const rows = items.filter(item => !item.docId?.endsWith?.('/index')).map(item => ({
        ...item,
        description: useDocById(item.docId ?? undefined)?.description,
    }));
    return toTable(
        [
            {title: "Label", formatter: (item) => <a href={
                item.href
            }>{item.label}</a>},
            {title: "Description", formatter: (item) => truncate(item.description ?? "", 70)},
        ],
        rows,
    );
}