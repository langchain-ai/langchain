# PII Redaction Middleware

The PII Redaction Middleware provides automatic detection and redaction of personally identifiable information (PII) from messages before they are sent to model providers, while restoring original values in model responses for tool execution.

## Features

- **Automatic PII Detection**: Uses regex patterns to identify PII in message content
- **Model Provider Isolation**: Redacts PII from requests to model providers
- **Tool Execution Restoration**: Restores original PII values for tool execution
- **Multiple Message Types**: Supports HumanMessage, AIMessage, SystemMessage, and ToolMessage
- **Structured Output Support**: Handles PII in tool calls and structured responses
- **Configurable Rules**: Customizable regex patterns for different PII types

## Usage

### Basic Setup

```python
import re
from langchain.agents.middleware.pii_redaction import PIIRedactionMiddleware
from langchain.agents import create_agent

# Define PII detection rules
PII_RULES = {
    "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
}

# Create agent with PII redaction middleware
agent = create_agent(
    model="openai:gpt-4",
    tools=[your_tools],
    middleware=[PIIRedactionMiddleware(rules=PII_RULES)]
)
```

### Runtime Configuration

```python
# Configure rules at runtime via middleware context
result = await agent.invoke(
    {"messages": [HumanMessage("My SSN is 123-45-6789")]},
    {
        "configurable": {
            "PIIRedactionMiddleware": {
                "rules": {
                    "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
                    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
                }
            }
        }
    }
)
```

## How It Works

### Request Phase (`modify_model_request`)
1. Scans all message content for PII using regex patterns
2. Replaces detected PII with redaction markers: `[REDACTED_{RULE_NAME}_{ID}]`
3. Stores original values in a redaction map
4. Returns modified request with redacted content

### Response Phase (`after_model`)
1. Scans model responses for redaction markers
2. Replaces markers with original values from redaction map
3. Handles both standard responses and structured output
4. Returns updated messages with restored PII

## Data Flow Example

```
User Input: "My SSN is 123-45-6789"
    ↓ [modify_model_request]
Model Request: "My SSN is [REDACTED_SSN_abc123]"
    ↓ [model invocation]
Model Response: tool_call({ "ssn": "[REDACTED_SSN_abc123]" })
    ↓ [after_model]
Tool Execution: tool({ "ssn": "123-45-6789" })
```

## Supported PII Types

The middleware can detect and redact various types of PII:

- **Social Security Numbers**: `123-45-6789`, `123456789`
- **Email Addresses**: `user@example.com`
- **Phone Numbers**: `(555) 123-4567`, `555-123-4567`
- **Credit Card Numbers**: `1234 5678 9012 3456`
- **Custom Patterns**: Any regex pattern you define

## Limitations

This middleware provides model provider isolation only. PII may still be present in:

- LangGraph state checkpoints (memory, databases)
- Network traffic between client and application server
- Application logs and trace data
- Tool execution arguments and responses
- Final agent output

For comprehensive PII protection, implement additional controls at the application, network, and storage layers.

## Security Considerations

- **Regex Patterns**: Ensure your regex patterns are comprehensive and tested
- **Redaction Map**: The redaction map is stored in memory and may persist across requests
- **Tool Execution**: Original PII values are restored for tool execution
- **Logging**: Be aware that PII may appear in application logs before redaction

## Examples

See `examples/pii_redaction_middleware_example.py` for a complete working example.
