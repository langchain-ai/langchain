from datetime import datetime
from typing import Optional

from langchain_core.prompts import PromptTemplate

from langchain_cohere.react_multi_hop.default_prompt_constants import (
    default_basic_rules,
    default_multi_hop_instruction,
    default_safety_rules,
    default_style_guide,
    default_system_prefix,
    default_task_context,
)


def render_structured_preamble(
    preamble: Optional[str] = None,
) -> str:
    if preamble is None:
        default_preamble = """## Task And Context
{task_and_context}

## Style Guide
{style_guide}"""
        preamble = default_preamble.format(
            task_and_context=default_task_context.format(
                now=datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")
            ),
            style_guide=default_style_guide,
        )

    structured_preamble_template = """{system_prefix}# Safety Preamble
{safety_rules}

# System Preamble
## Basic Rules
{basic_rules}

# User Preamble
{preamble}"""
    return structured_preamble_template.format(
        system_prefix=default_system_prefix,
        safety_rules=default_safety_rules,
        basic_rules=default_basic_rules,
        preamble=preamble,
    )


multi_hop_prompt_partial = PromptTemplate.from_template(
    """{structured_preamble}

## Available Tools
Here is a list of tools that you have available to you:

{tools}<|END_OF_TURN_TOKEN|>{history}{user_prompt}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{multi_hop_instruction}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{steps}"""
).partial(
    multi_hop_instruction=default_multi_hop_instruction,
)
