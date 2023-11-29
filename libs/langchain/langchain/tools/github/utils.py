from __future__ import annotations

from typing import TYPE_CHECKING

from github import GithubException
from github.Issue import Issue

from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

if TYPE_CHECKING:
    from github.Issue import Issue
    from langchain.chat_models.base import BaseChatModel


def generate_branch_name(issue: Issue, llm: BaseChatModel=None):
    """
    Helper functions. Use `generate_branch_name()` to generate a meaningful 
    branch name that the Agent will use to commit it's new code against. 
    Later, it can use this branch to open a pull request.
    """
    if not llm: 
        llm = ChatOpenAI(temperature=0, model="gpt-4")

    system_template = "You are a helpful assistant that writes clear and concise" \
                      "GitHub branch names for new pull requests."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    example_issue = {
        "title": "Implement an Integral function in C",
        "body": "This function should take as input a mathematical function " \
              "and the limits of integration and return the integral value.",
    }

    prompt = HumanMessagePromptTemplate.from_template(
        "Given this issue, please return a single string that would be a suitable " \
        "branch name on which to implement this feature request. Use common software " \
        "development best practices to name the branch.\n" \
        "Follow this formatting exactly:" \
        "Issue: {example_issue}" \
        "Branch name: `add_integral_in_c`\n\n" \
        "Issue: {issue}" \
        "Branch name: `" 
    )

    # Combine into a Chat conversation
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, prompt])
    formatted_messages = chat_prompt.format_messages(
        issue=str(issue), example_issue=str(example_issue)
    )

    output = llm(formatted_messages)
    return _ensure_unique_branch_name(
        issue.repository, _sanitize_branch_name(output.content)
    )


def _sanitize_branch_name(text):
    """
    # Remove non-alphanumeric characters, use underscores.
    Example:
        cleaned_text = strip_n_clean_text("Hello, World! This is an example.")
        print(cleaned_text)  # Output: "Hello_World_This_is_an_example"

    Returns:
        str: cleaned_text
    """
    cleaned_words = [
        "".join(c for c in word if c.isalnum()
                or c in ["_", "-"]) for word in text.split()
    ]
    return "_".join(cleaned_words)


def _ensure_unique_branch_name(repo, proposed_branch_name):
    # Attempt to create the branch, appending _v{i} if the name already exists
    i = 0
    new_branch_name = proposed_branch_name
    base_branch = repo.get_branch(repo.default_branch)
    for i in range(1000):
        try:
            repo.create_git_ref(
                ref=f"refs/heads/{new_branch_name}", sha=base_branch.commit.sha
            )
            print(f"Branch '{new_branch_name}' created successfully!")
            return new_branch_name
        except GithubException as e:
            if e.status == 422 and "Reference already exists" in e.data["message"]:
                i += 1
                new_branch_name = f"{proposed_branch_name}_v{i}"
                print(f"Branch name already exists. Trying with {new_branch_name}...")
            else:
                # Handle any other exceptions
                print(f"Failed to create branch. Error: {e}")
                raise Exception(
                    "Unable to create branch name from "
                    f"proposed_branch_name: {proposed_branch_name}")
    raise Exception(
        f"Unable to create branch. At least 1000 branches exist with"
        f"named derived from proposed_branch_name: `{proposed_branch_name}`"
    )
