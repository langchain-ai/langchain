# Security Policy

LangChain has a large ecosystem of integrations with various external resources like local and remote file systems, APIs and databases. These integrations allow developers to create versatile applications that combine the power of LLMs with the ability to access, interact with and manipulate external resources.

## Best practices

When building such applications, developers should remember to follow good security practices:

* [**Limit Permissions**](https://en.wikipedia.org/wiki/Principle_of_least_privilege): Scope permissions specifically to the application's need. Granting broad or excessive permissions can introduce significant security vulnerabilities. To avoid such vulnerabilities, consider using read-only credentials, disallowing access to sensitive resources, using sandboxing techniques (such as running inside a container), specifying proxy configurations to control external requests, etc., as appropriate for your application.
* **Anticipate Potential Misuse**: Just as humans can err, so can Large Language Models (LLMs). Always assume that any system access or credentials may be used in any way allowed by the permissions they are assigned. For example, if a pair of database credentials allows deleting data, it's safest to assume that any LLM able to use those credentials may in fact delete data.
* [**Defense in Depth**](https://en.wikipedia.org/wiki/Defense_in_depth_(computing)): No security technique is perfect. Fine-tuning and good chain design can reduce, but not eliminate, the odds that a Large Language Model (LLM) may make a mistake. It's best to combine multiple layered security approaches rather than relying on any single layer of defense to ensure security. For example: use both read-only permissions and sandboxing to ensure that LLMs are only able to access data that is explicitly meant for them to use.
* **Input Validation and Sanitization**: Implement strict validation and sanitization of all user inputs to prevent prompt injection attacks. Consider using input filtering, output encoding, and other techniques to guard against malicious inputs.
* **Auditing and Logging**: Maintain detailed logs of LLM decision processes, tool usage, and system access. Regularly audit these logs to detect anomalous behavior.

Risks of not doing so include, but are not limited to:

* Data corruption or loss.
* Unauthorized access to confidential information.
* Compromised performance or availability of critical resources.
* Supply chain attacks through malicious tools or integrations
* Prompt injection leading to unauthorized action execution
* Model theft or model parameter exfiltration

Example scenarios with mitigation strategies:

* A user may ask an agent with access to the file system to delete files that should not be deleted or read the content of files that contain sensitive information. To mitigate, limit the agent to only use a specific directory and only allow it to read or write files that are safe to read or write. Consider further sandboxing the agent by running it in a container.
* A user may ask an agent with write access to an external API to write malicious data to the API, or delete data from that API. To mitigate, give the agent read-only API keys, or limit it to only use endpoints that are already resistant to such misuse.
* A user may ask an agent with access to a database to drop a table or mutate the schema. To mitigate, scope the credentials to only the tables that the agent needs to access and consider issuing READ-ONLY credentials.
* A user may craft malicious prompts to bypass safety controls and execute unauthorized actions through an agent. To mitigate, implement robust input validation, use structured prompt templates, clearly separate user input from system instructions, and consider using prompt injection detection mechanisms.
* An agent may inadvertently leak sensitive information from its training data or system details in responses to user queries. To mitigate, implement output filtering and content moderation, use data masking techniques for sensitive information, and avoid including confidential data in prompts or context.

If you're building applications that access external resources like file systems, APIs or databases, consider speaking with your company's security team to determine how to best design and secure your applications.

## Reporting OSS Vulnerabilities

LangChain is partnered with [huntr by Protect AI](https://huntr.com/) to provide
a bounty program for our open source projects.

Please report security vulnerabilities associated with the LangChain
open source projects at [huntr](https://huntr.com/bounties/disclose/?target=https%3A%2F%2Fgithub.com%2Flangchain-ai%2Flangchain&validSearch=true).

Before reporting a vulnerability, please review:

1) In-Scope Targets and Out-of-Scope Targets below.
2) The [langchain-ai/langchain](https://docs.langchain.com/oss/python/contributing/code#repository-structure) monorepo structure.
3) The [Best Practices](#best-practices) above to understand what we consider to be a security vulnerability vs. developer responsibility.

### In-Scope Targets

The following packages and repositories are eligible for bug bounties:

* langchain-core
* langchain (see exceptions)
* langchain-community (see exceptions)
* langgraph
* langserve

### Out of Scope Targets

All out of scope targets defined by huntr as well as:

* **langchain-experimental**: This repository is for experimental code and is not
  eligible for bug bounties (see [package warning](https://pypi.org/project/langchain-experimental/)), bug reports to it will be marked as interesting or waste of
  time and published with no bounty attached.
* **tools**: Tools in either `langchain` or `langchain-community` are not eligible for bug
  bounties. This includes the following directories
  * `libs/langchain/langchain/tools`
  * `libs/community/langchain_community/tools`
  * Please review the [Best Practices](#best-practices)
    for more details, but generally tools interact with the real world. Developers are
    expected to understand the security implications of their code and are responsible
    for the security of their tools.
* Code documented with security notices. This will be decided on a case-by-case basis, but likely will not be eligible for a bounty as the code is already
  documented with guidelines for developers that should be followed for making their
  application secure.
* Any LangSmith related repositories or APIs (see [Reporting LangSmith Vulnerabilities](#reporting-langsmith-vulnerabilities)).

## Reporting LangSmith Vulnerabilities

Please report security vulnerabilities associated with LangSmith by email to `security@langchain.dev`.

* LangSmith site: [https://smith.langchain.com](https://smith.langchain.com)
* SDK client: [https://github.com/langchain-ai/langsmith-sdk](https://github.com/langchain-ai/langsmith-sdk)

### Other Security Concerns

For any other security concerns, please contact us at `security@langchain.dev`.
