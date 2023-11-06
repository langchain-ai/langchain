def test_default_code() -> None:
    with open("../langchain_cli/project_template/app/server.py", "r") as f:
        server_code = f.read()

    out_code = add_route_code(
        server_code,
    )

    # add
