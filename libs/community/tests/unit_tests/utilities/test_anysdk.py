import json

from langchain_community.utilities.anysdk import AnySdkWrapper, CrudControls


class FakeSdk:
    def get_thing(self, thing_id: int) -> dict:
        """Gets Things"""
        return self._dummy_return(thing_id)

    def create_thing(self, thing_id: int) -> dict:
        """Creates Things"""
        return self._dummy_return(thing_id)

    def post_thing(self, thing_id: int) -> dict:
        """Posts Things"""
        return self._dummy_return(thing_id)

    def put_thing(self, thing_id: int) -> dict:
        """Puts Things"""
        return self._dummy_return(thing_id)

    def delete_thing(self, thing_id: int) -> dict:
        """Deletes Things"""
        return self._dummy_return(thing_id)

    def _dummy_return(self, thing_id: int) -> dict:
        """Hidden, never called directly. Does Things"""
        return {"status": 200, "response": {"id": thing_id}}


client = {"client": FakeSdk()}

crud_controls = CrudControls(
    create=True,
    update=True,
    delete=True,
)

anysdk = AnySdkWrapper(
    client=client,
)


def test_operations_is_populated() -> None:
    assert len(anysdk.operations) != 0


def test_get_thing() -> None:
    assert (
        json.loads(anysdk.run("get_thing", json.dumps({"thing_id": 123})))["response"][
            "id"
        ]
        == 123
    )


def test_create_thing() -> None:
    assert (
        json.loads(anysdk.run("create_thing", json.dumps({"thing_id": 123})))[
            "response"
        ]["id"]
        == 123
    )


def test_post_thing() -> None:
    assert (
        json.loads(anysdk.run("post_thing", json.dumps({"thing_id": 123})))["response"][
            "id"
        ]
        == 123
    )


def test_put_thing() -> None:
    assert (
        json.loads(anysdk.run("put_thing", json.dumps({"thing_id": 123})))["response"][
            "id"
        ]
        == 123
    )


def test_delete_thing() -> None:
    assert (
        json.loads(anysdk.run("delete_thing", json.dumps({"thing_id": 123})))[
            "response"
        ]["id"]
        == 123
    )


def test_no_hidden_methods() -> None:
    assert not any(op["mode"].startswith("_") for op in anysdk.operations)
