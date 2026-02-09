"""Integration tests for the deepagents `SandboxClient` abstraction.

Implementers should subclass this test suite and provide a fixture that returns a
clean `SandboxClient` instance.

The provider is expected to support:

- `create()` creating a sandbox
- `get(sandbox_id=...)` reconnecting to an existing sandbox without creating a new one
- `delete()` being idempotent

Example:
```python
from __future__ import annotations

import pytest
from deepagents.backends.sandbox import SandboxClient
from langchain_tests.integration_tests import SandboxClientIntegrationTests

from langchain_acme_sandbox import AcmeSandboxClient


class TestAcmeSandboxClientStandard(SandboxClientIntegrationTests):
    @pytest.fixture
    def sandbox_provider(self) -> SandboxClient:
        # Return a provider instance in a clean state.
        return AcmeSandboxClient(api_key="...")

    @property
    def has_async(self) -> bool:
        return True
```

"""

# ruff: noqa: E402, S108

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import pytest

deepagents = pytest.importorskip("deepagents")

from deepagents.backends.protocol import (
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)

try:
    from deepagents.backends.sandbox import (
        SandboxClient,
        SandboxError,
        SandboxNotFoundError,
    )
except ImportError:  # pragma: no cover
    from typing import Any as SandboxClient

    class SandboxError(Exception):
        pass

    class SandboxNotFoundError(SandboxError):
        pass


from langchain_tests.base import BaseStandardTests

if TYPE_CHECKING:
    from collections.abc import Iterator


class SandboxClientIntegrationTests(BaseStandardTests):
    """Base class for sandbox client integration tests."""

    @pytest.fixture(scope="class")
    def sandbox_backend(
        self,
        sandbox_provider: SandboxClient,
    ) -> Iterator[SandboxBackendProtocol]:
        """Create one sandbox backend for the test class and clean it up at the end."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        backend = sandbox_provider.create()
        backend.execute("rm -rf /tmp/test_sandbox_ops && mkdir -p /tmp/test_sandbox_ops")
        try:
            yield backend
        finally:
            sandbox_provider.delete(sandbox_id=backend.id)

    @abstractmethod
    @pytest.fixture
    def sandbox_provider(self) -> SandboxClient:
        """Get a clean `SandboxClient` instance."""

    @property
    def has_sync(self) -> bool:
        """Configurable property to enable or disable sync tests."""
        return True

    @property
    def has_async(self) -> bool:
        """Configurable property to enable or disable async tests."""
        return True

    @property
    def supports_distinct_download_errors(self) -> bool:
        return True

    def test_create_then_get_then_delete_smoke(
        self,
        sandbox_provider: SandboxClient,
    ) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        backend = sandbox_provider.create()
        assert isinstance(backend.id, str)
        created_id = backend.id

        reconnected = sandbox_provider.get(sandbox_id=created_id)
        assert reconnected.id == created_id

        sandbox_provider.delete(sandbox_id=created_id)

    def test_execute_smoke(self, sandbox_backend: SandboxBackendProtocol) -> None:
        """Test that a sandbox can execute a basic command."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        result = sandbox_backend.execute("echo hello")
        assert result.output.strip() == "hello"

    def test_get_existing_does_not_create_new(
        self,
        sandbox_provider: SandboxClient,
    ) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        backend = sandbox_provider.create()
        created_id = backend.id

        try:
            _reconnected = sandbox_provider.get(sandbox_id=created_id)
            assert _reconnected.id == created_id
        finally:
            sandbox_provider.delete(sandbox_id=created_id)

    def test_get_missing_id_raises(
        self,
        sandbox_provider: SandboxClient,
    ) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        missing_id = "definitely-not-a-real-sandbox-id"
        with pytest.raises(SandboxNotFoundError):
            sandbox_provider.get(sandbox_id=missing_id)

        try:
            sandbox_provider.get(sandbox_id=missing_id)
        except SandboxNotFoundError:
            pass
        except SandboxError as e:
            msg = f"Expected SandboxNotFoundError, got SandboxError: {type(e).__name__}"
            raise AssertionError(msg) from e

    def test_delete_is_idempotent(self, sandbox_provider: SandboxClient) -> None:
        """Test `delete()` is idempotent and safe to call for missing IDs."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        backend = sandbox_provider.create()
        created_id = backend.id

        sandbox_provider.delete(sandbox_id=created_id)
        sandbox_provider.delete(sandbox_id=created_id)
        sandbox_provider.delete(sandbox_id="definitely-not-a-real-sandbox-id")

    async def test_async_create_then_delete_smoke(
        self,
        sandbox_provider: SandboxClient,
    ) -> None:
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        backend = await sandbox_provider.acreate()
        assert isinstance(backend.id, str)
        created_id = backend.id

        await sandbox_provider.adelete(sandbox_id=created_id)

    async def test_async_execute_smoke(
        self,
        sandbox_backend: SandboxBackendProtocol,
    ) -> None:
        """Async: test that a sandbox can execute a basic command."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        result = await sandbox_backend.aexecute("echo hello")
        assert result.output.strip() == "hello"

    async def test_async_upload_single_file(
        self,
        sandbox_backend: SandboxBackendProtocol,
    ) -> None:
        """Async: test uploading a single file."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        test_path = "/tmp/test_upload_single_async.txt"
        test_content = b"Hello, Sandbox!"

        upload_responses = await sandbox_backend.aupload_files([(test_path, test_content)])

        assert len(upload_responses) == 1
        assert upload_responses[0].path == test_path
        assert upload_responses[0].error is None

        result = await sandbox_backend.aexecute(f"cat {test_path}")
        assert result.output.strip() == test_content.decode()

    async def test_async_download_single_file(
        self,
        sandbox_backend: SandboxBackendProtocol,
    ) -> None:
        """Async: test downloading a single file."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        test_path = "/tmp/test_download_single_async.txt"
        test_content = b"Download test content"

        await sandbox_backend.aupload_files([(test_path, test_content)])

        download_responses = await sandbox_backend.adownload_files([test_path])

        assert len(download_responses) == 1
        assert download_responses[0].path == test_path
        assert download_responses[0].content == test_content
        assert download_responses[0].error is None

    async def test_async_upload_download_roundtrip(
        self,
        sandbox_backend: SandboxBackendProtocol,
    ) -> None:
        """Async: test upload followed by download for data integrity."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        test_path = "/tmp/test_roundtrip_async.txt"
        test_content = b"Roundtrip test: special chars \n\t\r\x00"

        upload_responses = await sandbox_backend.aupload_files([(test_path, test_content)])
        assert upload_responses[0].error is None

        download_responses = await sandbox_backend.adownload_files([test_path])
        assert download_responses[0].error is None
        assert download_responses[0].content == test_content


class SandboxIntegrationTests(BaseStandardTests):
    @pytest.fixture(scope="class")
    def sandbox_backend(
        self,
        sandbox_provider: SandboxClient,
    ) -> Iterator[SandboxBackendProtocol]:
        backend = sandbox_provider.create()
        backend.execute("rm -rf /tmp/test_sandbox_ops && mkdir -p /tmp/test_sandbox_ops")
        try:
            yield backend
        finally:
            sandbox_provider.delete(sandbox_id=backend.id)

    @property
    def supports_distinct_download_errors(self) -> bool:
        return True

    @abstractmethod
    @pytest.fixture
    def sandbox_provider(self) -> SandboxClient: ...

    @property
    def has_sync(self) -> bool:
        return True

    @property
    def has_async(self) -> bool:
        return True

    @pytest.fixture(autouse=True)
    def _setup_test_dir(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        sandbox_backend.execute("rm -rf /tmp/test_sandbox_ops && mkdir -p /tmp/test_sandbox_ops")

    def test_write_new_file(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        test_path = "/tmp/test_sandbox_ops/new_file.txt"
        content = "Hello, sandbox!\nLine 2\nLine 3"
        result = sandbox_backend.write(test_path, content)
        assert result.error is None
        assert result.path == test_path
        exec_result = sandbox_backend.execute(f"cat {test_path}")
        assert exec_result.output.strip() == content

    def test_read_basic_file(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        test_path = "/tmp/test_sandbox_ops/read_test.txt"
        content = "Line 1\nLine 2\nLine 3"
        sandbox_backend.write(test_path, content)
        result = sandbox_backend.read(test_path)
        assert "Error:" not in result
        assert all(line in result for line in ("Line 1", "Line 2", "Line 3"))

    def test_edit_single_occurrence(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        test_path = "/tmp/test_sandbox_ops/edit_single.txt"
        content = "Hello world\nGoodbye world\nHello again"
        sandbox_backend.write(test_path, content)
        result = sandbox_backend.edit(test_path, "Goodbye", "Farewell")
        assert result.error is None
        assert result.occurrences == 1
        file_content = sandbox_backend.read(test_path)
        assert "Farewell world" in file_content
        assert "Goodbye" not in file_content

    def test_ls_info_lists_files(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        sandbox_backend.write("/tmp/test_sandbox_ops/a.txt", "a")
        sandbox_backend.write("/tmp/test_sandbox_ops/b.txt", "b")
        info = sandbox_backend.ls_info("/tmp/test_sandbox_ops")
        paths = sorted([i["path"] for i in info])
        assert "/tmp/test_sandbox_ops/a.txt" in paths
        assert "/tmp/test_sandbox_ops/b.txt" in paths

    def test_glob_info(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        sandbox_backend.write("/tmp/test_sandbox_ops/x.py", "print('x')")
        sandbox_backend.write("/tmp/test_sandbox_ops/y.txt", "y")
        matches = sandbox_backend.glob_info("*.py", path="/tmp/test_sandbox_ops")
        assert [m["path"] for m in matches] == ["x.py"]

    def test_grep_raw_literal(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        sandbox_backend.write("/tmp/test_sandbox_ops/grep.txt", "a (b)\nstr | int\n")
        matches = sandbox_backend.grep_raw("str | int", path="/tmp/test_sandbox_ops")
        assert isinstance(matches, list)
        assert matches[0]["path"].endswith("/grep.txt")
        assert matches[0]["text"].strip() == "str | int"

    def test_upload_single_file(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = "/tmp/test_upload_single.txt"
        test_content = b"Hello, Sandbox!"

        upload_responses = sandbox_backend.upload_files([(test_path, test_content)])

        assert len(upload_responses) == 1
        assert upload_responses[0].path == test_path
        assert upload_responses[0].error is None

        result = sandbox_backend.execute(f"cat {test_path}")
        assert result.output.strip() == test_content.decode()

    def test_download_single_file(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = "/tmp/test_download_single.txt"
        test_content = b"Download test content"

        sandbox_backend.upload_files([(test_path, test_content)])

        download_responses = sandbox_backend.download_files([test_path])

        assert len(download_responses) == 1
        assert download_responses[0].path == test_path
        assert download_responses[0].content == test_content
        assert download_responses[0].error is None

    def test_upload_download_roundtrip(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = "/tmp/test_roundtrip.txt"
        test_content = b"Roundtrip test: special chars \n\t\r\x00"

        upload_responses = sandbox_backend.upload_files([(test_path, test_content)])
        assert upload_responses == [FileUploadResponse(path=test_path, error=None)]

        download_responses = sandbox_backend.download_files([test_path])
        assert download_responses == [FileDownloadResponse(path=test_path, content=test_content, error=None)]

    def test_upload_multiple_files_order_preserved(
        self,
        sandbox_backend: SandboxBackendProtocol,
    ) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        files = [
            ("/tmp/test_multi_1.txt", b"Content 1"),
            ("/tmp/test_multi_2.txt", b"Content 2"),
            ("/tmp/test_multi_3.txt", b"Content 3"),
        ]

        upload_responses = sandbox_backend.upload_files(files)

        assert upload_responses == [
            FileUploadResponse(path=files[0][0], error=None),
            FileUploadResponse(path=files[1][0], error=None),
            FileUploadResponse(path=files[2][0], error=None),
        ]

    def test_download_multiple_files_order_preserved(
        self,
        sandbox_backend: SandboxBackendProtocol,
    ) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        files = [
            ("/tmp/test_batch_1.txt", b"Batch 1"),
            ("/tmp/test_batch_2.txt", b"Batch 2"),
            ("/tmp/test_batch_3.txt", b"Batch 3"),
        ]
        sandbox_backend.upload_files(files)

        paths = [p for p, _ in files]
        download_responses = sandbox_backend.download_files(paths)

        assert download_responses == [
            FileDownloadResponse(path=files[0][0], content=files[0][1], error=None),
            FileDownloadResponse(path=files[1][0], content=files[1][1], error=None),
            FileDownloadResponse(path=files[2][0], content=files[2][1], error=None),
        ]

    def test_upload_binary_content_roundtrip(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = "/tmp/binary_file.bin"
        test_content = bytes(range(256))

        upload_responses = sandbox_backend.upload_files([(test_path, test_content)])
        assert upload_responses == [FileUploadResponse(path=test_path, error=None)]

        download_responses = sandbox_backend.download_files([test_path])
        assert download_responses == [FileDownloadResponse(path=test_path, content=test_content, error=None)]

    def test_download_error_file_not_found(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        missing_path = "/tmp/nonexistent_test_file.txt"

        responses = sandbox_backend.download_files([missing_path])

        assert responses == [
            FileDownloadResponse(path=missing_path, content=None, error="file_not_found")
        ]

    def test_download_error_is_directory(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        dir_path = "/tmp/test_directory"
        sandbox_backend.execute(f"rm -rf {dir_path} && mkdir -p {dir_path}")

        responses = sandbox_backend.download_files([dir_path])

        if not self.supports_distinct_download_errors:
            assert responses == [
                FileDownloadResponse(path=dir_path, content=None, error="file_not_found")
            ]
            return

        assert responses == [
            FileDownloadResponse(path=dir_path, content=None, error="is_directory")
        ]

    def test_download_error_permission_denied(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = "/tmp/test_no_read.txt"
        sandbox_backend.execute(
            f"rm -f {test_path} && echo secret > {test_path} && chmod 000 {test_path}"
        )

        try:
            responses = sandbox_backend.download_files([test_path])
        finally:
            sandbox_backend.execute(f"chmod 644 {test_path} || true")

        if not self.supports_distinct_download_errors:
            assert responses == [
                FileDownloadResponse(path=test_path, content=None, error="file_not_found")
            ]
            return

        assert responses == [
            FileDownloadResponse(path=test_path, content=None, error="permission_denied")
        ]

    def test_download_error_invalid_path_relative(
        self,
        sandbox_backend: SandboxBackendProtocol,
    ) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        responses = sandbox_backend.download_files(["relative/path.txt"])

        assert responses == [
            FileDownloadResponse(
                path="relative/path.txt",
                content=None,
                error="invalid_path",
            )
        ]

    def test_upload_missing_parent_dir_or_roundtrip(
        self,
        sandbox_backend: SandboxBackendProtocol,
    ) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        dir_path = "/tmp/test_upload_missing_parent_dir"
        path = f"{dir_path}/deepagents_test_upload.txt"
        content = b"nope"
        sandbox_backend.execute(f"rm -rf {dir_path}")

        responses = sandbox_backend.upload_files([(path, content)])
        assert len(responses) == 1
        assert responses[0].path == path

        if responses[0].error is not None:
            assert responses[0].error in {
                "invalid_path",
                "permission_denied",
                "file_not_found",
            }
            return

        download = sandbox_backend.download_files([path])
        assert download == [FileDownloadResponse(path=path, content=content, error=None)]

    def test_upload_relative_path_returns_invalid_path(
        self,
        sandbox_backend: SandboxBackendProtocol,
    ) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        path = "relative_upload.txt"
        content = b"nope"
        responses = sandbox_backend.upload_files([(path, content)])

        assert responses == [FileUploadResponse(path=path, error="invalid_path")]

    def test_get_existing_does_not_create_new(
        self,
        sandbox_provider: SandboxClient,
    ) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        backend = sandbox_provider.create()
        created_id = backend.id

        try:
            _reconnected = sandbox_provider.get(sandbox_id=created_id)
            assert _reconnected.id == created_id
        finally:
            sandbox_provider.delete(sandbox_id=created_id)

    def test_get_missing_id_raises(
        self,
        sandbox_provider: SandboxClient,
    ) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        missing_id = "definitely-not-a-real-sandbox-id"
        with pytest.raises(SandboxNotFoundError):
            sandbox_provider.get(sandbox_id=missing_id)

        try:
            sandbox_provider.get(sandbox_id=missing_id)
        except SandboxNotFoundError:
            pass
        except SandboxError as e:
            msg = f"Expected SandboxNotFoundError, got SandboxError: {type(e).__name__}"
            raise AssertionError(msg) from e

    def test_delete_is_idempotent(self, sandbox_provider: SandboxClient) -> None:
        """Test `delete()` is idempotent and safe to call for missing IDs."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        backend = sandbox_provider.create()
        created_id = backend.id

        sandbox_provider.delete(sandbox_id=created_id)
        sandbox_provider.delete(sandbox_id=created_id)
        sandbox_provider.delete(sandbox_id="definitely-not-a-real-sandbox-id")

    async def test_async_get_existing_does_not_create_new(
        self,
        sandbox_provider: SandboxClient,
    ) -> None:
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        backend = await sandbox_provider.acreate()
        created_id = backend.id

        try:
            _reconnected = await sandbox_provider.aget(sandbox_id=created_id)
            assert _reconnected.id == created_id
        finally:
            await sandbox_provider.adelete(sandbox_id=created_id)

    async def test_async_get_or_create_missing_id_raises(
        self,
        sandbox_provider: SandboxClient,
    ) -> None:
        """Async: test missing sandbox_id raises a `SandboxNotFoundError`."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        missing_id = "definitely-not-a-real-sandbox-id"
        with pytest.raises(SandboxNotFoundError):
            await sandbox_provider.aget(sandbox_id=missing_id)

        try:
            await sandbox_provider.aget(sandbox_id=missing_id)
        except SandboxNotFoundError:
            pass
        except SandboxError as e:
            msg = f"Expected SandboxNotFoundError, got SandboxError: {type(e).__name__}"
            raise AssertionError(msg) from e

    async def test_async_delete_is_idempotent(
        self,
        sandbox_provider: SandboxClient,
    ) -> None:
        """Async: test `delete()` is idempotent and safe to call for missing IDs."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        backend = await sandbox_provider.acreate()
        created_id = backend.id

        await sandbox_provider.adelete(sandbox_id=created_id)
        await sandbox_provider.adelete(sandbox_id=created_id)
        await sandbox_provider.adelete(sandbox_id="definitely-not-a-real-sandbox-id")
