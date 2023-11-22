import json
from typing import Iterable
from urllib.parse import urljoin

import pytest
import responses

from langchain.pydantic_v1 import SecretStr
from langchain.utilities.cube import CubeAPIWrapper

CUBE_API_URL = "http://cube-api:4000"
CUBE_API_TOKEN = "TOKEN"
CUBES = [
    {
        "name": "Users",
        "title": "Users",
        "connectedComponent": 1,
        "measures": [
            {
                "name": "users.count",
                "title": "Users Count",
                "shortTitle": "Count",
                "aliasName": "users.count",
                "type": "number",
                "aggType": "count",
                "drillMembers": ["users.id", "users.city", "users.createdAt"],
            }
        ],
        "dimensions": [
            {
                "name": "users.city",
                "title": "Users City",
                "type": "string",
                "aliasName": "users.city",
                "shortTitle": "City",
                "suggestFilterValues": True,
            }
        ],
        "segments": [],
    },
    {
        "name": "Orders",
        "title": "Orders",
        "connectedComponent": 1,
        "measures": [
            {
                "name": "orders.count",
                "title": "Orders Count",
                "shortTitle": "Count",
                "aliasName": "orders.count",
                "type": "number",
                "aggType": "count",
                "drillMembers": ["orders.id", "orders.type", "orders.createdAt"],
            }
        ],
        "dimensions": [
            {
                "name": "orders.type",
                "title": "Orders Type",
                "type": "string",
                "aliasName": "orders.city",
                "shortTitle": "Type",
                "suggestFilterValues": True,
            }
        ],
        "segments": [],
    },
]


@pytest.fixture(autouse=True)
def mocked_responses() -> Iterable[responses.RequestsMock]:
    """Fixture mocking requests.get."""
    with responses.RequestsMock() as rsps:
        yield rsps


def mocked_mata(mocked_responses: responses.RequestsMock) -> None:
    mocked_responses.add(
        method=responses.GET,
        url=urljoin(CUBE_API_URL, "/meta"),
        body=json.dumps({"cubes": CUBES}),
    )


def test_meta_information(mocked_responses: responses.RequestsMock) -> None:
    mocked_mata(mocked_responses)

    cube = CubeAPIWrapper(
        cube_api_url=CUBE_API_URL,
        cube_api_token=SecretStr(CUBE_API_TOKEN),
    )

    assert cube.meta_information == CUBES


def test_model_meta_information(mocked_responses: responses.RequestsMock) -> None:
    mocked_mata(mocked_responses)

    cube = CubeAPIWrapper(
        cube_api_url=CUBE_API_URL,
        cube_api_token=SecretStr(CUBE_API_TOKEN),
    )

    model_meta_information = """## Model: Orders

### Measures:
| Title | Description | Column | Type |
| --- | --- | --- | --- |
| Count |  | orders.count | number |

### Dimensions:
| Title | Description | Column | Type |
| --- | --- | --- | --- |
| Type |  | orders.type | string |


## Model: Users

### Measures:
| Title | Description | Column | Type |
| --- | --- | --- | --- |
| Count |  | users.count | number |

### Dimensions:
| Title | Description | Column | Type |
| --- | --- | --- | --- |
| City |  | users.city | string |
"""

    assert cube.model_meta_information == model_meta_information


def test_get_model_meta_information(mocked_responses: responses.RequestsMock) -> None:
    mocked_mata(mocked_responses)

    cube = CubeAPIWrapper(
        cube_api_url=CUBE_API_URL,
        cube_api_token=SecretStr(CUBE_API_TOKEN),
    )

    model_meta_information = """## Model: Users

### Measures:
| Title | Description | Column | Type |
| --- | --- | --- | --- |
| Count |  | users.count | number |

### Dimensions:
| Title | Description | Column | Type |
| --- | --- | --- | --- |
| City |  | users.city | string |
"""

    assert cube.get_model_meta_information(["Users"]) == model_meta_information


def test_get_usable_model_names(mocked_responses: responses.RequestsMock) -> None:
    mocked_mata(mocked_responses)

    cube = CubeAPIWrapper(
        cube_api_url=CUBE_API_URL,
        cube_api_token=SecretStr(CUBE_API_TOKEN),
    )

    assert cube.get_usable_model_names() == ["Orders", "Users"]


def test_get_usable_model_names_with_ignore_models(
    mocked_responses: responses.RequestsMock
) -> None:
    mocked_mata(mocked_responses)

    cube = CubeAPIWrapper(
        cube_api_url=CUBE_API_URL,
        cube_api_token=SecretStr(CUBE_API_TOKEN),
        ignore_models=["Users"],
    )

    assert cube.get_usable_model_names() == ["Orders"]


def test_get_usable_model_names_with_include_models(
    mocked_responses: responses.RequestsMock
) -> None:
    mocked_mata(mocked_responses)

    cube = CubeAPIWrapper(
        cube_api_url=CUBE_API_URL,
        cube_api_token=SecretStr(CUBE_API_TOKEN),
        include_models=["Users"],
    )

    assert cube.get_usable_model_names() == ["Users"]


def test_get_usable_models(mocked_responses: responses.RequestsMock) -> None:
    mocked_mata(mocked_responses)

    cube = CubeAPIWrapper(
        cube_api_url=CUBE_API_URL,
        cube_api_token=SecretStr(CUBE_API_TOKEN),
    )

    usable_models = """| Model | Description |
| --- | --- |
| Users | Users |
| Orders | Orders |
"""

    assert cube.get_usable_models() == usable_models


def test_get_usable_models_with_ignore_models(
    mocked_responses: responses.RequestsMock
) -> None:
    mocked_mata(mocked_responses)

    cube = CubeAPIWrapper(
        cube_api_url=CUBE_API_URL,
        cube_api_token=SecretStr(CUBE_API_TOKEN),
        ignore_models=["Users"],
    )

    usable_models = """| Model | Description |
| --- | --- |
| Orders | Orders |
"""

    assert cube.get_usable_models() == usable_models


def test_get_usable_models_with_include_models(
    mocked_responses: responses.RequestsMock
) -> None:
    mocked_mata(mocked_responses)

    cube = CubeAPIWrapper(
        cube_api_url=CUBE_API_URL,
        cube_api_token=SecretStr(CUBE_API_TOKEN),
        include_models=["Users"],
    )

    usable_models = """| Model | Description |
| --- | --- |
| Users | Users |
"""

    assert cube.get_usable_models() == usable_models


def test_get_model_meta_information_with_custom_model_info(
    mocked_responses: responses.RequestsMock
) -> None:
    mocked_mata(mocked_responses)

    users = """## Model: Users

### Measures:
| Title | Description | Column | Type |
| --- | --- | --- | --- |
| Count | User Count~~ | users.count | number |

### Dimensions:
| Title | Description | Column | Type |
| --- | --- | --- | --- |
| City | User City~~ | users.city | string |
| Address | User Address~~ | users.address | string |
"""

    cube = CubeAPIWrapper(
        cube_api_url=CUBE_API_URL,
        cube_api_token=SecretStr(CUBE_API_TOKEN),
        custom_model_info={
            "Users": users,
        },
    )

    model_meta_information = """## Model: Orders

### Measures:
| Title | Description | Column | Type |
| --- | --- | --- | --- |
| Count |  | orders.count | number |

### Dimensions:
| Title | Description | Column | Type |
| --- | --- | --- | --- |
| Type |  | orders.type | string |


## Model: Users

### Measures:
| Title | Description | Column | Type |
| --- | --- | --- | --- |
| Count | User Count~~ | users.count | number |

### Dimensions:
| Title | Description | Column | Type |
| --- | --- | --- | --- |
| City | User City~~ | users.city | string |
| Address | User Address~~ | users.address | string |
"""

    assert cube.get_model_meta_information() == model_meta_information
