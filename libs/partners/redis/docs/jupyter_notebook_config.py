# mypy: disable-error-code="name-defined"
c = get_config()  # type: ignore # noqa: F821
c.FileContentsManager.delete_to_trash = False
c.FileCheckpoints.checkpoint_dir = "/tmp/checkpoints"
