# Security Policy

## Reporting OSS Vulnerabilities

LangChain is partnered with [huntr by Protect AI](https://huntr.com/) to provide 
a bounty program for our open source projects. 

Please report security vulnerabilities associated with the LangChain 
open source projects by visiting the following link:

[https://huntr.com/bounties/disclose/](https://huntr.com/bounties/disclose/?target=https%3A%2F%2Fgithub.com%2Flangchain-ai%2Flangchain&validSearch=true)

Before reporting a vulnerability, please review:

1) In-Scope Targets and Out-of-Scope Targets below.
2) The [langchain-ai/langchain](https://python.langchain.com/docs/contributing/repo_structure) monorepo structure.
3) LangChain [security guidelines](https://python.langchain.com/docs/security) to
   understand what we consider to be a security vulnerability vs. developer
   responsibility.

### In-Scope Targets

The following packages and repositories are eligible for bug bounties:

- langchain-core
- langchain (see exceptions)
- langchain-community (see exceptions)
- langgraph
- langserve

### Out of Scope Targets

All out of scope targets defined by huntr as well as:

- **langchain-experimental**: This repository is for experimental code and is not
  eligible for bug bounties, bug reports to it will be marked as interesting or waste of
  time and published with no bounty attached.
- **tools**: Tools in either langchain or langchain-community are not eligible for bug
  bounties. This includes the following directories
  - langchain/tools
  - langchain-community/tools
  - Please review our [security guidelines](https://python.langchain.com/docs/security)
    for more details, but generally tools interact with the real world. Developers are
    expected to understand the security implications of their code and are responsible
    for the security of their tools.
- Code documented with security notices. This will be decided done on a case by
  case basis, but likely will not be eligible for a bounty as the code is already
  documented with guidelines for developers that should be followed for making their
  application secure.
- Any LangSmith related repositories or APIs see below.

## Reporting LangSmith Vulnerabilities

Please report security vulnerabilities associated with LangSmith by email to `security@langchain.dev`.

- LangSmith site: https://smith.langchain.com
- SDK client: https://github.com/langchain-ai/langsmith-sdk

### Other Security Concerns

For any other security concerns, please contact us at `security@langchain.dev`.
