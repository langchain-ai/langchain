# Security Policy

LangChain has a large ecosystem of integrations with various external resources like local and remote file systems, APIs and databases. These integrations allow developers to create versatile applications that combine the power of LLMs with the ability to access, interact with and manipulate external resources.

## Best practices

When building such applications developers should remember to follow good security practices:

* [**Limit Permissions**](https://en.wikipedia.org/wiki/Principle_of_least_privilege): Scope permissions specifically to the application's need. Granting broad or excessive permissions can introduce significant security vulnerabilities. To avoid such vulnerabilities, consider using read-only credentials, disallowing access to sensitive resources, using sandboxing techniques (such as running inside a container), specifying proxy configurations to control external requests, etc. as appropriate for your application.
* **Anticipate Potential Misuse**: Just as humans can err, so can Large Language Models (LLMs). Always assume that any system access or credentials may be used in any way allowed by the permissions they are assigned. For example, if a pair of database credentials allows deleting data, it's safest to assume that any LLM able to use those credentials may in fact delete data.
* [**Defense in Depth**](https://en.wikipedia.org/wiki/Defense_in_depth_(computing)): No security technique is perfect. Fine-tuning and good chain design can reduce, but not eliminate, the odds that a Large Language Model (LLM) may make a mistake. It's best to combine multiple layered security approaches rather than relying on any single layer of defense to ensure security. For example: use both read-only permissions and sandboxing to ensure that LLMs are only able to access data that is explicitly meant for them to use.

Risks of not doing so include, but are not limited to:
* Data corruption or loss.
* Unauthorized access to confidential information.
* Compromised performance or availability of critical resources.

Example scenarios with mitigation strategies:

* A user may ask an agent with access to the file system to delete files that should not be deleted or read the content of files that contain sensitive information. To mitigate, limit the agent to only use a specific directory and only allow it to read or write files that are safe to read or write. Consider further sandboxing the agent by running it in a container.
* A user may ask an agent with write access to an external API to write malicious data to the API, or delete data from that API. To mitigate, give the agent read-only API keys, or limit it to only use endpoints that are already resistant to such misuse.
* A user may ask an agent with access to a database to drop a table or mutate the schema. To mitigate, scope the credentials to only the tables that the agent needs to access and consider issuing READ-ONLY credentials.

If you're building applications that access external resources like file systems, APIs
or databases, consider speaking with your company's security team to determine how to best
design and secure your applications.

## API Key & Credential Safety

Do not pass API keys, tokens, or secrets directly through LLM prompts or outputs. LLMs may inadvertently expose these in responses or logs.

Best practices:
- Store secrets using environment variables or secret managers
- Avoid embedding credentials in chains or tools unless scoped and encrypted
- Monitor logs to ensure sensitive info is not being passed into LLM calls

## Threat Modeling for LLM Applications

When designing an LLM-powered system, consider these common threat surfaces:

- **Input Surface**: Can users provide prompts or instructions? If so, how are inputs validated?
- **Output Surface**: Can LLM outputs trigger actions (e.g., file writes, API calls)? Are these outputs filtered or moderated?
- **Contextual Leakage**: Does the system store prompts, memory, or internal reasoning? Could this be leaked or tampered with?
- **Tool Use**: If your agents can invoke tools, what guardrails prevent misuse?

A structured threat model helps determine where to apply input sanitization, rate limits, logging, and fallback controls.

## Prompt Injection

Prompt injection is a specific threat to LLM-powered applications where users craft inputs designed to override instructions, leak internal data, or trigger unintended tool usage.

LangChain integrates with third-party defenses (e.g., ZenGuard), but developers should also adopt core mitigation strategies during prompt construction.

### Common Risks:
- Overriding system prompts via user input
- Triggering unintended agent or tool actions
- Leaking internal reasoning or configuration logic
- Causing hallucinations that result in unsafe behavior

### Mitigation Strategies:

- **Use `PromptTemplate`** to safely isolate user inputs.

  **Example**

  Instead of this (vulnerable to injection):
  ```python
  prompt = f"Tell me about {user_input}"

  Use this:
  from langchain_core.prompts import PromptTemplate

  template = PromptTemplate.from_template("Tell me about {topic}")
  prompt = template.format(topic=user_input)

- **Avoid raw string interpolation** with `f""` or `.format()` for prompts
- **Apply runtime guards**, such as:
  - [ZenGuard](https://github.com/zenml-io/zenml)
- **Rate-limit LLM endpoints**
- **Log and monitor prompt interactions**

For broader LLM guidance, see the [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

## Memory & Context Leakage

LangChain supports memory modules that store past interactions (e.g., `ConversationBufferMemory`). When using memory:

- Always scope memory to individual users or sessions
- Avoid storing personally identifiable or sensitive data unless necessary
- Regularly flush or expire stale memory entries

## Reporting OSS Vulnerabilities

LangChain is partnered with [huntr by Protect AI](https://huntr.com/) to provide 
a bounty program for our open source projects. 

Please report security vulnerabilities associated with the LangChain 
open source projects by visiting the following link:

[https://huntr.com/bounties/disclose/](https://huntr.com/bounties/disclose/?target=https%3A%2F%2Fgithub.com%2Flangchain-ai%2Flangchain&validSearch=true)

Before reporting a vulnerability, please review:

1) In-Scope Targets and Out-of-Scope Targets below.
2) The [langchain-ai/langchain](https://python.langchain.com/docs/contributing/repo_structure) monorepo structure.
3) The [Best practices](#best-practices) above to
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
  eligible for bug bounties (see [package warning](https://pypi.org/project/langchain-experimental/)), bug reports to it will be marked as interesting or waste of
  time and published with no bounty attached.
- **tools**: Tools in either langchain or langchain-community are not eligible for bug
  bounties. This includes the following directories
  - libs/langchain/langchain/tools
  - libs/community/langchain_community/tools
  - Please review the [best practices](#best-practices)
    for more details, but generally tools interact with the real world. Developers are
    expected to understand the security implications of their code and are responsible
    for the security of their tools.
- Code documented with security notices. This will be decided done on a case by
  case basis, but likely will not be eligible for a bounty as the code is already
  documented with guidelines for developers that should be followed for making their
  application secure.
- Any LangSmith related repositories or APIs (see [Reporting LangSmith Vulnerabilities](#reporting-langsmith-vulnerabilities)).

## Reporting LangSmith Vulnerabilities

Please report security vulnerabilities associated with LangSmith by email to `security@langchain.dev`.

- LangSmith site: https://smith.langchain.com
- SDK client: https://github.com/langchain-ai/langsmith-sdk

### Other Security Concerns

For any other security concerns, please contact us at `security@langchain.dev`.

