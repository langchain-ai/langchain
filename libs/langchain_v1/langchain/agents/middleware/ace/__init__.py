"""ACE (Agentic Context Engineering) middleware for self-improving agents.

This middleware implements the ACE framework which enables agents to self-improve
by treating contexts as evolving playbooks that accumulate, refine, and organize
strategies through a modular process of generation, reflection, and curation.

Example:
    ```python
    from langchain.agents import create_agent
    from langchain.agents.middleware import ACEMiddleware

    # Create ACE middleware (both models are required)
    ace = ACEMiddleware(
        reflector_model="gpt-4o-mini",
        curator_model="gpt-4o-mini",
        curator_frequency=10,
    )

    # Create agent with ACE middleware
    agent = create_agent(
        model="gpt-4o",
        tools=[...],
        middleware=[ace],
    )

    # The agent will self-improve through playbook evolution
    result = agent.invoke({"messages": [HumanMessage(content="...")]})
    ```

For more information, see:
- Paper: https://arxiv.org/abs/2510.04618
- Documentation: https://docs.langchain.com/oss/python/langchain/middleware
"""

from langchain.agents.middleware.ace.middleware import ACEMiddleware
from langchain.agents.middleware.ace.playbook import (
    ACEPlaybook,
    SECTION_NAMES,
    extract_bullet_ids,
    format_playbook_line,
    get_playbook_stats,
    initialize_empty_playbook,
    parse_playbook_line,
    update_bullet_counts,
)

__all__ = [
    "ACEMiddleware",
    "ACEPlaybook",
    "SECTION_NAMES",
    "extract_bullet_ids",
    "format_playbook_line",
    "get_playbook_stats",
    "initialize_empty_playbook",
    "parse_playbook_line",
    "update_bullet_counts",
]
