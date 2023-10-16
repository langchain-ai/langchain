# Security

LangChain has a large ecosystem of integrations with various external resources like local and remote file systems, APIs and databases. These integrations allow developers to create versatile applications that combine the power of LLMs with the ability to access, interact with and manipulate external resources.

## Best Practices

When building such applications developers should remember to follow good security practices:

* Limit Permissions: Scope permissions specifically to the application's need. Granting broad or excessive permissions can introduce significant security vulnerabilities. When relevant, consider further sandboxing the code appropriately (e.g., run it inside a container).
* Account for Errors: Just as humans can err, so can Large Language Models (LLMs). Such errors might lead to unintended consequences.
* Anticipate Potential Misuse: Always assume that the credentials may be used in any way allowed by the permissions they are assigned.

Risks of not doing so include, but are not limited to:
* Data corruption or loss.
* Unauthorized access to confidential information.
* Compromising performance or availability of external resources.

Example scenarios with mitigation strategies:

* A user may ask an agent with access to the file system to delete files that should not be deleted or read the content of files that contain sensitive information. To mitigate, limit the agent to only use a specific directory and only allow it to read or write files that are safe to read or write. Consider further sandboxing the agent by running it in a container.
* A user may ask an agent with access to an external API with write permissions to write malicious data to the API. To mitigate, only allow the agent to interact with endpoints that are safe to use.
* A user may ask an agent with access to a database to drop a table or mutate the schema. To mitigate, scope the credentials to only the tables that the agent needs to access and consider issuing READ-ONLY credentials.

If you're building applications that access external resources like file systems, APIs
or databases, consider speaking with your company's security team to determine how to best
design and secure your applications.

## Reporting a Vulnerability

Please report security vulnerabilities by email to security@langchain.dev. This email is
an alias to a subset of our maintainers, and will ensure the issue is promptly triaged
and acted upon as needed.

## Roadmap

Over the next few months, we are planning on breaking apart the LangChain package
into smaller packages to separate out core functionality (e.g., schema, callbacks, LLMs) 
from integrations that manipulate external resources (e.g., database access, file management etc.).

This should make it easier for developers and the security teams in their organizations
to manage and assess the security of their applications.
