# flake8: noqa
GET_ISSUES_PROMPT = """
This tool will fetch a list of the repository's issues. It will return the title, and issue number of 5 issues. It takes no input.
"""

GET_ISSUE_PROMPT = """
This tool will fetch the title, body, and comment thread of a specific issue. **VERY IMPORTANT**: You must specify the issue number as an integer.
"""

COMMENT_ON_ISSUE_PROMPT = """
This tool is useful when you need to comment on a GitLab issue. Simply pass in the issue number and the comment you would like to make. Please use this sparingly as we don't want to clutter the comment threads. **VERY IMPORTANT**: Your input to this tool MUST strictly follow these rules:

- First you must specify the issue number as an integer
- Then you must place two newlines
- Then you must specify your comment
"""
CREATE_PULL_REQUEST_PROMPT = """
This tool is useful when you need to create a new pull request in a GitLab repository. **VERY IMPORTANT**: Your input to this tool MUST strictly follow these rules:

- First you must specify the title of the pull request
- Then you must place two newlines
- Then you must write the body or description of the pull request

To reference an issue in the body, put its issue number directly after a #.
For example, if you would like to create a pull request called "README updates" with contents "added contributors' names, closes issue #3", you would pass in the following string:

README updates

added contributors' names, closes issue #3
"""
CREATE_FILE_PROMPT = """
This tool is a wrapper for the GitLab API, useful when you need to create a file in a GitLab repository. **VERY IMPORTANT**: Your input to this tool MUST strictly follow these rules:

- First you must specify which file to create by passing a full file path (**IMPORTANT**: the path must not start with a slash)
- Then you must specify the contents of the file

For example, if you would like to create a file called /test/test.txt with contents "test contents", you would pass in the following string:

test/test.txt

test contents
"""

READ_FILE_PROMPT = """
This tool is a wrapper for the GitLab API, useful when you need to read the contents of a file in a GitLab repository. Simply pass in the full file path of the file you would like to read. **IMPORTANT**: the path must not start with a slash
"""

UPDATE_FILE_PROMPT = """
This tool is a wrapper for the GitLab API, useful when you need to update the contents of a file in a GitLab repository. **VERY IMPORTANT**: Your input to this tool MUST strictly follow these rules:

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
>>>> NEW
"""

DELETE_FILE_PROMPT = """
This tool is a wrapper for the GitLab API, useful when you need to delete a file in a GitLab repository. Simply pass in the full file path of the file you would like to delete. **IMPORTANT**: the path must not start with a slash
"""

GET_REPO_FILES_IN_MAIN = """
This tool will provide an overview of all existing files in the main branch of the GitLab repository repository. It will list the file names. No input parameters are required.
"""

GET_REPO_FILES_IN_BOT_BRANCH = """
This tool will provide an overview of all files in your current working branch where you should implement changes. No input parameters are required.
"""

GET_REPO_FILES_FROM_DIRECTORY = """
This tool will provide an overview of all files in your current working branch from a specific directory. **VERY IMPORTANT**: You must specify the path of the directory as a string input parameter.
"""

LIST_REPO_BRANCES = """
This tool is a wrapper for the GitLab API, useful when you need to read the branches names in a GitLab repository. No input parameters are required.
"""

CREATE_REPO_BRANCH = """
This tool will create a new branch in the repository. **VERY IMPORTANT**: You must specify the name of the new branch as a string input parameter.
"""

SET_ACTIVE_BRANCH = """
This tool will set the active branch in the repository, similar to `git checkout <branch_name>` and `git switch -c <branch_name>`. **VERY IMPORTANT**: You must specify the name of the branch as a string input parameter.
"""
