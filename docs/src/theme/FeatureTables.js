import React from "react";

const FeatureTables = {
    llms: {
        link: "/docs/integrations/llms",
        items:[
            {
                "Model": "Anthropic",
                "link": "anthropic.ipynb",
                "package": "langchain-anthropic",
            }
        ]
    },
    text_embedding: {
        link: "/docs/integrations/text_embedding",
        items:[
            {
                "Model": "Cohere",
                "link": "cohere.ipynb",
                "package": "langchain-cohere",
            }
        ]
    },
};

function formatItem(item: object) {
    const rtn = {...item};
    // 
}

function to2dArray(items: object[]) {
    const keys = Object.keys(items[0]);

    return items.map((item) => Object.values(item));
}

export function CategoryTable() {
    return (<table><thead><tr><th>Model</th></tr></thead></table>);
}

export function ItemTable() {
    return <div>Item Table</div>;
}