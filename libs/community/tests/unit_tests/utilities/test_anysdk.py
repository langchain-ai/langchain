import json

from langchain_community.utilities.anysdk import AnySdkWrapper, CrudControls


class FakeSdk:
    def get_things(self) -> dict:
        """Gets Thing"""
        return self._dummy_return(123)

    def get_thing(self, thing_id: int) -> dict:
        """Gets Thing"""
        return self._dummy_return(thing_id)

    def create_thing(self, thing_id: int) -> dict:
        """Creates Thing"""
        return self._dummy_return(thing_id)

    def post_thing(self, thing_id: int) -> dict:
        """Posts Thing"""
        return self._dummy_return(thing_id)

    def put_thing(self, thing_id: int) -> dict:
        """Puts Thing"""
        return self._dummy_return(thing_id)

    def delete_thing(self, thing_id: int) -> dict:
        """Deletes Thing"""
        return self._dummy_return(thing_id)

    def confabulate_thing(self, thing_id: int) -> dict:
        """Confabulates Thing -- example of custom verbs"""
        return self._dummy_return(thing_id)

    def _dummy_return(self, thing_id: int) -> dict:
        """Hidden, never called directly. Does Things"""
        return {"status": 200, "response": {"id": thing_id}}


client = {"client": FakeSdk()}
crud_controls: CrudControls = CrudControls(
    read=True,
    create=True,
    update=True,
    delete=True,
)

anysdk = AnySdkWrapper(
    client=client,
    crud_controls=crud_controls,  # type: ignore
)


def test_operations_is_populated() -> None:
    assert len(anysdk.operations) != 0


def test_get_things_no_input() -> None:
    matching_tool = next(
        (tool for tool in anysdk.operations if tool.name == "get_things"), None
    )
    if matching_tool:
        assert json.loads(matching_tool._run())["response"]["id"] == 123
    else:
        raise ValueError("No matching tool found.")


def test_get_thing() -> None:
    matching_tool = next(
        (tool for tool in anysdk.operations if tool.name == "get_thing"), None
    )
    if matching_tool:
        assert (
            json.loads(matching_tool._run(json.dumps({"thing_id": 123})))["response"][
                "id"
            ]
            == 123
        )
    else:
        raise ValueError("No matching tool found.")


def test_create_thing_string_input() -> None:
    matching_tool = next(
        (tool for tool in anysdk.operations if tool.name == "create_thing"), None
    )
    if matching_tool:
        assert (
            json.loads(matching_tool._run(json.dumps({"thing_id": 123})))["response"][
                "id"
            ]
            == 123
        )
    else:
        raise ValueError("No matching tool found.")


def test_create_thing_kwarg_input() -> None:
    matching_tool = next(
        (tool for tool in anysdk.operations if tool.name == "create_thing"), None
    )
    if matching_tool:
        assert json.loads(matching_tool._run(thing_id=123))["response"]["id"] == 123  # type: ignore
    else:
        raise ValueError("No matching tool found.")


def test_post_thing() -> None:
    matching_tool = next(
        (tool for tool in anysdk.operations if tool.name == "post_thing"), None
    )
    if matching_tool:
        assert (
            json.loads(matching_tool._run(json.dumps({"thing_id": 123})))["response"][
                "id"
            ]
            == 123
        )
    else:
        raise ValueError("No matching tool found.")


def test_put_thing() -> None:
    matching_tool = next(
        (tool for tool in anysdk.operations if tool.name == "put_thing"), None
    )
    if matching_tool:
        assert (
            json.loads(matching_tool._run(json.dumps({"thing_id": 123})))["response"][
                "id"
            ]
            == 123
        )
    else:
        raise ValueError("No matching tool found.")


def test_delete_thing() -> None:
    matching_tool = next(
        (tool for tool in anysdk.operations if tool.name == "delete_thing"), None
    )
    if matching_tool:
        assert (
            json.loads(matching_tool.run(json.dumps({"thing_id": 123})))["response"][
                "id"
            ]
            == 123
        )
    else:
        raise ValueError("No matching tool found.")


def test_confabulate_thing() -> None:
    """tests example sdk customization"""
    crud_controls = CrudControls(
        read_list="confabulate",
    )

    anysdk = AnySdkWrapper(
        client=client,
        crud_controls=crud_controls,  # type: ignore
    )
    matching_tool = next(
        (tool for tool in anysdk.operations if tool.name == "confabulate_thing"), None
    )
    if matching_tool:
        assert (
            json.loads(matching_tool.run(json.dumps({"thing_id": 123})))["response"][
                "id"
            ]
            == 123
        )
    else:
        raise ValueError("No matching tool found.")


def test_no_hidden_methods() -> None:
    assert not any([op.name.startswith("_") for op in anysdk.operations])
