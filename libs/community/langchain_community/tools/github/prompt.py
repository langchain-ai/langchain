# flake8: noqa
GET_ISSUES_PROMPT = """
This tool will fetch a list of the repository's issues. It will return the title, and issue number of 5 issues. It takes no input."""

GET_ISSUE_PROMPT = """
This tool will fetch the title, body, and comment thread of a specific issue. **VERY IMPORTANT**: You must specify the issue number as an integer."""

COMMENT_ON_ISSUE_PROMPT = """
This tool is useful when you need to comment on a GitHub issue. Simply pass in the issue number and the comment you would like to make. Please use this sparingly as we don't want to clutter the comment threads. **VERY IMPORTANT**: Your input to this tool MUST strictly follow these rules:

- First you must specify the issue number as an integer
- Then you must place two newlines
- Then you must specify your comment"""

CREATE_PULL_REQUEST_PROMPT = """
This tool is useful when you need to create a new pull request in a GitHub repository. **VERY IMPORTANT**: Your input to this tool MUST strictly follow these rules:

- First you must specify the title of the pull request
- Then you must place two newlines
- Then you must write the body or description of the pull request

When appropriate, always reference relevant issues in the body by using the syntax `closes #<issue_number` like `closes #3, closes #6`.
For example, if you would like to create a pull request called "README updates" with contents "added contributors' names, closes #3", you would pass in the following string:

README updates

added contributors' names, closes #3"""

CREATE_FILE_PROMPT = """
This tool is a wrapper for the GitHub API, useful when you need to create a file in a GitHub repository. **VERY IMPORTANT**: Your input to this tool MUST strictly follow these rules:

- First you must specify which file to create by passing a full file path (**IMPORTANT**: the path must not start with a slash)
- Then you must specify the contents of the file

For example, if you would like to create a file called /test/test.txt with contents "test contents", you would pass in the following string:

test/test.txt

test contents"""

READ_FILE_PROMPT = """
This tool is a wrapper for the GitHub API, useful when you need to read the contents of a file. Simply pass in the full file path of the file you would like to read. **IMPORTANT**: the path must not start with a slash"""

UPDATE_FILE_PROMPT = """
This tool is a wrapper for the GitHub API, useful when you need to update the contents of a file in a GitHub repository. **VERY IMPORTANT**: Your input to this tool MUST strictly follow these rules:

- First you must specify which file to modify by passing a full file path (**IMPORTANT**: the path must not start with a slash)
- Then you must specify the old contents which you would like to replace wrapped in OLD <<<< and >>>> OLD
- Then you must specify the new contents which you would like to replace the old contents with wrapped in NEW <<<< and >>>> NEW

For example, if you would like to replace the contents of the file /test/test.txt from "old contents" to "new contents", you would pass in the following string:

test/test.txt

This is text that will not be changed
OLD <<<<
old contents
>>>> OLD
NEW <<<<
new contents
>>>> NEW"""

DELETE_FILE_PROMPT = """
This tool is a wrapper for the GitHub API, useful when you need to delete a file in a GitHub repository. Simply pass in the full file path of the file you would like to delete. **IMPORTANT**: the path must not start with a slash"""

GET_PR_PROMPT = """
This tool will fetch the title, body, comment thread and commit history of a specific Pull Request (by PR number). **VERY IMPORTANT**: You must specify the PR number as an integer."""

LIST_PRS_PROMPT = """
This tool will fetch a list of the repository's Pull Requests (PRs). It will return the title, and PR number of 5 PRs. It takes no input."""

LIST_PULL_REQUEST_FILES = """
This tool will fetch the full text of all files in a pull request (PR) given the PR number as an input. This is useful for understanding the code changes in a PR or contributing to it. **VERY IMPORTANT**: You must specify the PR number as an integer input parameter."""

OVERVIEW_EXISTING_FILES_IN_MAIN = """
This tool will provide an overview of all existing files in the main branch of the repository. It will list the file names, their respective paths, and a brief summary of their contents. This can be useful for understanding the structure and content of the repository, especially when navigating through large codebases. No input parameters are required."""

OVERVIEW_EXISTING_FILES_BOT_BRANCH = """
This tool will provide an overview of all files in your current working branch where you should implement changes. This is great for getting a high level overview of the structure of your code. No input parameters are required."""

SEARCH_ISSUES_AND_PRS_PROMPT = """
This tool will search for issues and pull requests in the repository. **VERY IMPORTANT**: You must specify the search query as a string input parameter."""

SEARCH_CODE_PROMPT = """
This tool will search for code in the repository. **VERY IMPORTANT**: You must specify the search query as a string input parameter."""

CREATE_REVIEW_REQUEST_PROMPT = """
This tool will create a review request on the open pull request that matches the current active branch. **VERY IMPORTANT**: You must specify the username of the person who is being requested as a string input parameter."""

LIST_BRANCHES_IN_REPO_PROMPT = """
This tool will fetch a list of all branches in the repository. It will return the name of each branch. No input parameters are required."""

SET_ACTIVE_BRANCH_PROMPT = """
This tool will set the active branch in the repository, similar to `git checkout <branch_name>` and `git switch -c <branch_name>`. **VERY IMPORTANT**: You must specify the name of the branch as a string input parameter."""

CREATE_BRANCH_PROMPT = """
This tool will create a new branch in the repository. **VERY IMPORTANT**: You must specify the name of the new branch as a string input parameter."""

GET_FILES_FROM_DIRECTORY_PROMPT = """
This tool will fetch a list of all files in a specified directory. **VERY IMPORTANT**: You must specify the path of the directory as a string input parameter."""

GET_LATEST_RELEASE_PROMPT = """
This tool will fetch the latest release of the repository. No input parameters are required."""

GET_RELEASES_PROMPT = """
This tool will fetch the latest 5 releases of the repository. No input parameters are required."""

GET_RELEASE_PROMPT = """
This tool will fetch a specific release of the repository. **VERY IMPORTANT**: You must specify the tag name of the release as a string input parameter."""
