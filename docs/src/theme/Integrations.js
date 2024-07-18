import React from "react";

/*
| Type | Package | Source | Updates | Releases | Start Here | Maintenance Help | Integration Testing |
| ---- | ------- | ----------- | ------------- | ----------- | ------------------- | ------------------- |
| Community | [`langchain-community`](https://pypi.org/project/langchain-community) | [Monorepo](https://github.com/langchain-ai/langchain/tree/master/libs/community) | Community | LangChain | ✅ | ✅ | ❌ |
| Shared Packages | Separate | Shared Repo | Partner | Partner & LangChain | ❌ | ✅ | ✅ |
| Monorepo Packages | Separate | [Monorepo](https://github.com/langchain-ai/langchain/tree/master/libs/partners) | Partner | LangChain | ❌ | ✅ |✅ |
| External Packages | Separate | External Repo | Partner | Partner | ❌ | ❌ | ⚠️ |
*/

const INTEGRATION_TYPES = [
    {
        type: 'Community',
        package: <a href="https://pypi.org/project/langchain-community"><code>langchain-community</code></a>,
        source: (<a href="https://github.com/langchain-ai/langchain/tree/master/libs/community">Monorepo</a>),
        updates: 'Community',
        releases: 'LangChain',
        startHere: true,
        maintenanceHelp: true,
        integrationTesting: false,
    },
    {
        type: 'Shared Packages',
        package: (<code>langchain-[partner]</code>),
        source: 'Shared Repo',
        updates: 'Partner',
        releases: 'Partner & LangChain',
        startHere: false,
        maintenanceHelp: true,
        integrationTesting: true,
    },
    {
        type: 'Monorepo Packages',
        package: (<code>langchain-[partner]</code>),
        source: (<a href="https://github.com/langchain-ai/langchain/tree/master/libs/partners">Monorepo</a>),
        updates: 'Partner',
        releases: 'LangChain',
        startHere: false,
        maintenanceHelp: true,
        integrationTesting: true,
    },
    {
        type: 'External Packages',
        package: (<code>langchain-[partner]</code>),
        source: 'External Repo',
        updates: 'Partner',
        releases: 'Partner',
        startHere: false,
        maintenanceHelp: false,
        integrationTesting: '⚠️',
    },
]

function renderCell(cell) {
    if (typeof cell === 'boolean') {
        return cell ? '✅' : '❌'
    }
    return cell
}

export function IntegrationsTable() {
    const headings = [
        // { name: 'Type', key: 'type' },
        // { name: 'Package', key: 'package' },
        // { name: 'Updates', key: 'updates' },
        { name: 'Releases', key: 'releases' },
        { name: 'Maintenance Help', key: 'maintenanceHelp' },
        { name: 'Integration Testing', key: 'integrationTesting' },
    ]
    const transposed = headings.map(({ name, key }) => {
        const values = INTEGRATION_TYPES.map(integration => integration[key])
        return [name, ...values]
    }
    )
    return (
        <table style={{textAlign: 'center'}}>
            <tr>
                <th rowSpan={2}>Type</th>
                <th rowSpan={2}>Community</th>
                <th colspan={3}>Package in</th>
            </tr>
            <tr>
                <th>Shared repo</th>
                <th>LangChain repo</th>
                <th>External repo</th>
            </tr>
            <tr>
                <th>Package</th>
                <td><a href="https://pypi.org/project/langchain-community"><code>langchain-community</code></a></td>
                <td colSpan={3}><code>langchain-[partner]</code></td>
            </tr>
            <tr>
                <th>Updates</th>
                <td>Community</td>
                <td colSpan={3}>Partner</td>
            </tr>

            {transposed.map((row, i) => (
                <tr key={i}>
                    {row.map((cell, j) => (j === 0) ? (
                        <th key={j}>{renderCell(cell)}</th>
                    ) : (
                        <td key={j}>{renderCell(cell)}</td>
                    ))}
                </tr>
            ))}
        </table>
    )
}