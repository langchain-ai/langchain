"""
**Comprehend Moderation** is used to detect and handle `Personally Identifiable Information (PII)`,
`toxicity`, and `prompt safety` in text.

The Langchain experimental package includes the **AmazonComprehendModerationChain** class
for the comprehend moderation tasks. It is based on `Amazon Comprehend` service.
This class can be configured with specific moderation settings like PII labels, redaction,
toxicity thresholds, and prompt safety thresholds.

See more at https://aws.amazon.com/comprehend/

`Amazon Comprehend` service is used by several other classes:
- **ComprehendToxicity** class is used to check the toxicity of text prompts using
  `AWS Comprehend service` and take actions based on the configuration
- **ComprehendPromptSafety** class is used to validate the safety of given prompt
  text, raising an error if unsafe content is detected based on the specified threshold
- **ComprehendPII** class is designed to handle
  `Personally Identifiable Information (PII)` moderation tasks,
  detecting and managing PII entities in text inputs
"""  # noqa: E501

from langchain_experimental.comprehend_moderation.amazon_comprehend_moderation import (
    AmazonComprehendModerationChain,
)
from langchain_experimental.comprehend_moderation.base_moderation import BaseModeration
from langchain_experimental.comprehend_moderation.base_moderation_callbacks import (
    BaseModerationCallbackHandler,
)
from langchain_experimental.comprehend_moderation.base_moderation_config import (
    BaseModerationConfig,
    ModerationPiiConfig,
    ModerationPromptSafetyConfig,
    ModerationToxicityConfig,
)
from langchain_experimental.comprehend_moderation.pii import ComprehendPII
from langchain_experimental.comprehend_moderation.prompt_safety import (
    ComprehendPromptSafety,
)
from langchain_experimental.comprehend_moderation.toxicity import ComprehendToxicity

__all__ = [
    "BaseModeration",
    "ComprehendPII",
    "ComprehendPromptSafety",
    "ComprehendToxicity",
    "BaseModerationConfig",
    "ModerationPiiConfig",
    "ModerationToxicityConfig",
    "ModerationPromptSafetyConfig",
    "BaseModerationCallbackHandler",
    "AmazonComprehendModerationChain",
]
