import React from "react";

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
};

function toTable(columns: Column[], items: any[]) {
    const headers = columns.map((col) => col.title);
    return (
        <table>
            <thead>
                <tr>
                    {headers.map((header) => <th>{header}</th>)}
                </tr>
            </thead>
            <tbody>
                {items.map((item) => (
                    <tr>
                        {columns.map((col) => <td>{col.formatter(item)}</td>)}
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
