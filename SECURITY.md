# Security Policy

## Reporting OSS Vulnerabilities

LangChain is partnered with [huntr by Protect AI](https://huntr.com/) to provide a bounty program for our open source projects.

Please report security vulnerabilities associated with the LangChain open source projects by visiting the following link:

[https://huntr.com/bounties/disclose/](https://huntr.com/bounties/disclose/?target=https%3A%2F%2Fgithub.com%2Flangchain-ai%2Flangchain&validSearch=true)

Before reporting a vulnerability, please review the following:

1. **In-Scope Targets** and **Out-of-Scope Targets** listed below.
2. The [langchain-ai/langchain](https://python.langchain.com/docs/contributing/repo_structure) monorepo structure.
3. LangChain [security guidelines](https://python.langchain.com/docs/security) to understand what we consider a security vulnerability vs. developer responsibility.

### In-Scope Targets

The following packages and repositories are eligible for bug bounties:

- `langchain-core`
- `langchain` (with some exceptions)
- `langchain-community` (with some exceptions)
- `langgraph`
- `langserve`

### Out-of-Scope Targets

All targets deemed out-of-scope by huntr, as well as:

- **`langchain-experimental`**: This repository is for experimental code and is not eligible for bug bounties. Bug reports for this will be marked as "interesting" or "waste of time" and published with no bounty attached.
- **Tools**: Tools in either `langchain` or `langchain-community` are not eligible for bug bounties. This includes:
  - `langchain/tools`
  - `langchain-community/tools`
  - Please review our [security guidelines](https://python.langchain.com/docs/security) for more details. Generally, tools interact with the real world, and developers are responsible for understanding the security implications of their tools.
- Code documented with security notices: On a case-by-case basis, code with security guidelines may not be eligible for bounties. Such code is already documented, and developers should follow the guidelines to make their applications secure.
- Any LangSmith-related repositories or APIs (details below).

## Reporting LangSmith Vulnerabilities

Please report security vulnerabilities associated with LangSmith by email to `security@langchain.dev`.

- **LangSmith site**: [https://smith.langchain.com](https://smith.langchain.com)
- **SDK client**: [https://github.com/langchain-ai/langsmith-sdk](https://github.com/langchain-ai/langsmith-sdk)

### Other Security Concerns

For any other security concerns, please contact us at `security@langchain.dev`.
