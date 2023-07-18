# Dev container

This project includes a [dev container](https://containers.dev/), which lets you use a container as a full-featured dev environment.

You can use the dev container configuration in this folder to build and run the app without needing to install any of its tools locally! You can use it in [GitHub Codespaces](https://github.com/features/codespaces) or the [VS Code Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

## GitHub Codespaces
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/hwchase17/langchain)

You may use the button above, or follow these steps to open this repo in a Codespace:
1. Click the **Code** drop-down menu at the top of https://github.com/hwchase17/langchain.
1. Click on the **Codespaces** tab.
1. Click **Create codespace on master** .

For more info, check out the [GitHub documentation](https://docs.github.com/en/free-pro-team@latest/github/developing-online-with-codespaces/creating-a-codespace#creating-a-codespace).
  
## VS Code Dev Containers
[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/hwchase17/langchain)

If you already have VS Code and Docker installed, you can use the button above to get started. This will cause VS Code to automatically install the Dev Containers extension if needed, clone the source code into a container volume, and spin up a dev container for use.

You can also follow these steps to open this repo in a container using the VS Code Dev Containers extension:

1. If this is your first time using a development container, please ensure your system meets the pre-reqs (i.e. have Docker installed) in the [getting started steps](https://aka.ms/vscode-remote/containers/getting-started).

2. Open a locally cloned copy of the code:

   - Clone this repository to your local filesystem.
   - Press <kbd>F1</kbd> and select the **Dev Containers: Open Folder in Container...** command.
   - Select the cloned copy of this folder, wait for the container to start, and try things out!

You can learn more in the [Dev Containers documentation](https://code.visualstudio.com/docs/devcontainers/containers).

## Tips and tricks

* If you are working with the same repository folder in a container and Windows, you'll want consistent line endings (otherwise you may see hundreds of changes in the SCM view). The `.gitattributes` file in the root of this repo will disable line ending conversion and should prevent this. See [tips and tricks](https://code.visualstudio.com/docs/devcontainers/tips-and-tricks#_resolving-git-line-ending-issues-in-containers-resulting-in-many-modified-files) for more info.
* If you'd like to review the contents of the image used in this dev container, you can check it out in the [devcontainers/images](https://github.com/devcontainers/images/tree/main/src/python) repo.
