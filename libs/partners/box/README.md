# langchain-box

This package contains the LangChain integration with Box. For more information about
Box, check out our [developer documentation](https://developer.box.com).

## Pre-requisites

In order to integrate with Box, you need a few things:

* A Box instance — if you are not a current Box customer, sign up for a 
[free dev account](https://account.box.com/signup/n/developer#ty9l3).
* A Box app — more on how to 
[create an app](https://developer.box.com/guides/getting-started/first-application/)
* Your app approved in your Box instance — This is done by your admin.
The good news is if you are using a free developer account, you are the admin.
[Authorize your app](https://developer.box.com/guides/authorization/custom-app-approval/#manual-approval)

## Installation

```bash
pip install -U langchain-box
```

## Authentication

The `box-langchain` package offers some flexibility to authentication. The
most basic authentication method is by using a developer token. This can be
found in the [Box developer console](https://account.box.com/developers/console) 
on the configuration screen. This token is purposely short-lived (1 hour) and is 
intended for development. With this token, you can add it to your environment as 
`BOX_DEVELOPER_TOKEN`, you can pass it directly to the loader, or you can use the 
`BoxAuth` authentication helper class.

We will cover passing it directly to the loader in the section below. 

### BoxAuth helper class

`BoxAuth` supports the following authentication methods:

* Token — either a developer token or any token generated through the Box SDK
* JWT with a service account
* JWT with a specified user
* CCG with a service account
* CCG with a specified user

> [!NOTE]
> If using JWT authentication, you will need to download the configuration from the Box
> developer console after generating your public/private key pair. Place this file in your 
> application directory structure somewhere. You will use the path to this file when using
> the `BoxAuth` helper class.

For more information, learn about how to 
[set up a Box application](https://developer.box.com/guides/getting-started/first-application/),
and check out the 
[Box authentication guide](https://developer.box.com/guides/authentication/select/)
for more about our different authentication options.

Examples:

**Token**

```python
from langchain_box.document_loaders import BoxLoader
from langchain_box.utilities import BoxAuth, BoxAuthType

auth = BoxAuth(
    auth_type=BoxAuthType.TOKEN,
    box_developer_token=box_developer_token
)

loader = BoxLoader(
    box_auth=auth,
    ...
)
```

**JWT with a service account**

```python
from langchain_box.document_loaders import BoxLoader
from langchain_box.utilities import BoxAuth, BoxAuthType

auth = BoxAuth(
    auth_type=BoxAuthType.JWT,
    box_jwt_path=box_jwt_path
)

loader = BoxLoader(
    box_auth=auth,
    ...
```

**JWT with a specified user**

```python
from langchain_box.document_loaders import BoxLoader
from langchain_box.utilities import BoxAuth, BoxAuthType

auth = BoxAuth(
    auth_type=BoxAuthType.JWT,
    box_jwt_path=box_jwt_path,
    box_user_id=box_user_id
)

loader = BoxLoader(
    box_auth=auth,
    ...
```

**CCG with a service account**

```python
from langchain_box.document_loaders import BoxLoader
from langchain_box.utilities import BoxAuth, BoxAuthType

auth = BoxAuth(
    auth_type=BoxAuthType.CCG,
    box_client_id=box_client_id,
    box_client_secret=box_client_secret,
    box_enterprise_id=box_enterprise_id
)

loader = BoxLoader(
    box_auth=auth,
    ...
```

**CCG with a specified user**

```python
from langchain_box.document_loaders import BoxLoader
from langchain_box.utilities import BoxAuth, BoxAuthType

auth = BoxAuth(
    auth_type=BoxAuthType.CCG,
    box_client_id=box_client_id,
    box_client_secret=box_client_secret,
    box_user_id=box_user_id
)

loader = BoxLoader(
    box_auth=auth,
    ...
```

## Document Loaders

The `BoxLoader` class helps you get your unstructured content from Box
in Langchain's `Document` format. You can do this with either a `List[str]`
containing Box file IDs, or with a `str` containing a Box folder ID. 

If getting files from a folder with folder ID, you can also set a `Bool` to
tell the loader to get all sub-folders in that folder, as well. 

:::info
A Box instance can contain Petabytes of files, and folders can contain millions
of files. Be intentional when choosing what folders you choose to index. And we
recommend never getting all files from folder 0 recursively. Folder ID 0 is your
root folder.
:::

### Load files

```python
import os

from langchain_box.document_loaders import BoxLoader

os.environ["BOX_DEVELOPER_TOKEN"] = "df21df2df21df2d1f21df2df1"

loader = BoxLoader(
    box_file_ids=["12345", "67890"],
    character_limit=10000  # Optional. Defaults to no limit
)

docs = loader.lazy_load()
```

### Load from folder

```python
import os

from langchain_box.document_loaders import BoxLoader

os.environ["BOX_DEVELOPER_TOKEN"] = "df21df2df21df2d1f21df2df1"

loader = BoxLoader(
    box_folder_id="12345",
    recursive=False,  # Optional. return entire tree, defaults to False
    character_limit=10000  # Optional. Defaults to no limit
)

docs = loader.lazy_load()
```