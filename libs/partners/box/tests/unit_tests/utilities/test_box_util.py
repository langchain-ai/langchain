import pytest
from pydantic.v1.error_wrappers import ValidationError

from langchain_box.utilities import BoxAPIWrapper, BoxAuth, BoxAuthType


# Test auth types
def test_token_initialization() -> None:
    auth = BoxAuth(
        auth_type=BoxAuthType.TOKEN, box_developer_token="box_developer_token"
    )

    assert auth.auth_type == "token"
    assert auth.box_developer_token == "box_developer_token"


def test_failed_token_initialization() -> None:
    with pytest.raises(ValidationError):
        auth = BoxAuth(auth_type=BoxAuthType.TOKEN)  # noqa: F841


def test_jwt_eid_initialization() -> None:
    auth = BoxAuth(auth_type=BoxAuthType.JWT, box_jwt_path="box_jwt_path")

    assert auth.auth_type == "jwt"
    assert auth.box_jwt_path == "box_jwt_path"


def test_jwt_user_initialization() -> None:
    auth = BoxAuth(
        auth_type=BoxAuthType.JWT,
        box_jwt_path="box_jwt_path",
        box_user_id="box_user_id",
    )

    assert auth.auth_type == "jwt"
    assert auth.box_jwt_path == "box_jwt_path"
    assert auth.box_user_id == "box_user_id"


def test_failed_jwt_initialization() -> None:
    with pytest.raises(ValidationError):
        auth = BoxAuth(auth_type=BoxAuthType.JWT, box_user_id="box_user_id")  # noqa: F841


def test_ccg_eid_initialization() -> None:
    auth = BoxAuth(
        auth_type=BoxAuthType.CCG,
        box_client_id="box_client_id",
        box_client_secret="box_client_secret",
        box_enterprise_id="box_enterprise_id",
    )

    assert auth.auth_type == "ccg"
    assert auth.box_client_id == "box_client_id"
    assert auth.box_client_secret == "box_client_secret"
    assert auth.box_enterprise_id == "box_enterprise_id"


def test_ccg_user_initialization() -> None:
    auth = BoxAuth(
        auth_type=BoxAuthType.CCG,
        box_client_id="box_client_id",
        box_client_secret="box_client_secret",
        box_enterprise_id="box_enterprise_id",
        box_user_id="box_user_id",
    )

    assert auth.auth_type == "ccg"
    assert auth.box_client_id == "box_client_id"
    assert auth.box_client_secret == "box_client_secret"
    assert auth.box_enterprise_id == "box_enterprise_id"
    assert auth.box_user_id == "box_user_id"


def test_failed_ccg_initialization() -> None:
    with pytest.raises(ValidationError):
        auth = BoxAuth(auth_type=BoxAuthType.CCG)  # noqa: F841


def test_direct_token_initialization():
    box = BoxAPIWrapper(
        box_developer_token="box_developer_token",
        get_text_rep=True,
        box_get_images=False,
    )

    assert box.box_developer_token == "box_developer_token"
    assert box.get_text_rep is True
    assert box.get_images is False


def test_auth_initialization():
    auth = BoxAuth(
        auth_type=BoxAuthType.TOKEN, box_developer_token="box_developer_token"
    )

    box = BoxAPIWrapper(box_auth=auth, get_text_rep=True, box_get_images=False)

    assert auth.box_developer_token == "box_developer_token"
    assert box.get_text_rep is True
    assert box.get_images is False


def test_failed_initialization_no_auth() -> None:
    with pytest.raises(ValidationError):
        box = BoxAPIWrapper(get_text_rep=True, box_get_images=False)  # noqa: F841
