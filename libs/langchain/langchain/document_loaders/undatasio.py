"""UnDatasIO document loader."""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
from langchain_core.documents import Document

from langchain.document_loaders.base import BaseLoader

try:
    from undatasio import UnDatasIO
except ImportError as err:
    msg = "undatasio package not found. Please install it with `pip install undatasio`."
    raise ImportError(msg) from err


class UnDatasIOLoader(BaseLoader):
    """Load *parsed text* from UnDatasIO API (PDF/图片/OCR)."""

    def __init__(
        self,
        token: str,
        file_path: str,
        workspace_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """Initialize loader.

        Args:
            token: UnDatasIO API access token.
            file_path: File path to upload and parse.
            workspace_id: Target workspace (optional).
            task_id: Target task (optional).
        """
        self.client = UnDatasIO(token=token)
        self.file_path = file_path
        self.workspace_id = workspace_id
        self.task_id = task_id

    def _get_or_pick_workspace(self) -> str:
        if self.workspace_id:
            return self.workspace_id
        ws = self.client.workspace_list()
        if not ws:
            msg = "No workspace found"
            raise RuntimeError(msg)
        return ws[0]["work_id"]

    def _get_or_pick_task(self, work_id: str) -> str:
        if self.task_id:
            return self.task_id
        ts = self.client.task_list(work_id=work_id)
        if not ts:
            msg = "No task found"
            raise RuntimeError(msg)
        return ts[0]["task_id"]

    def load(self) -> list[Document]:
        """Upload -> parse -> return Document."""
        # 1. 选 workspace / task
        work_id = self._get_or_pick_workspace()
        task_id = self._get_or_pick_task(work_id)

        # 2. 上传
        if not self.client.upload_file(task_id=task_id, file_path=self.file_path):
            msg = "Upload failed"
            raise RuntimeError(msg)

        # 3. 找到刚上传的文件
        files = self.client.get_task_files(task_id=task_id)
        file_id = next(
            f["file_id"]
            for f in files
            if f["file_name"] == self.file_path.split("/")[-1]
        )

        # 4. 触发解析
        if not self.client.parse_files(task_id=task_id, file_ids=[file_id]):
            msg = "Parse trigger failed"
            raise RuntimeError(msg)

        # 5. 轮询状态
        while True:
            time.sleep(5)
            task_files = pd.DataFrame(self.client.get_task_files(task_id=task_id))
            status = task_files.loc[
                task_files["file_id"] == file_id, "status"
            ].to_numpy()[0]
            if status == "parser success":
                break
            if status in ("parser failed", "error"):
                msg = "Parsing failed"
                raise RuntimeError(msg)

        # 6. 拿文本
        text_lines = self.client.get_parse_result(task_id=task_id, file_id=file_id)
        if not text_lines:
            msg = "No parse result"
            raise RuntimeError(msg)
        text = "\n".join(text_lines)

        # 7. 返回 Document
        return [
            Document(
                page_content=text,
                metadata={
                    "source": self.file_path,
                    "task_id": task_id,
                    "file_id": file_id,
                },
            )
        ]
