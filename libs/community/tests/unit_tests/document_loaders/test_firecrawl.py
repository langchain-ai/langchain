"""Test FireCrawlLoader."""

import sys
from typing import Generator, List, Tuple
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders import FireCrawlLoader


# firecrawl 모듈을 모킹하여 sys.modules에 등록
@pytest.fixture(autouse=True)
def mock_firecrawl() -> Generator[Tuple[MagicMock, MagicMock], None, None]:
    """Mock firecrawl module for all tests."""
    mock_module = MagicMock()
    mock_client = MagicMock()
    # FirecrawlApp 클래스로 수정
    mock_module.FirecrawlApp.return_value = mock_client

    # extract 메서드의 반환값 설정
    response_dict = {
        "success": True,
        "data": {
            "title": "extracted title",
            "main contents": "extracted main contents",
        },
        "status": "completed",
        "expiresAt": "2025-03-12T12:42:09.000Z",
    }
    mock_client.extract.return_value = response_dict

    # sys.modules에 모의 모듈 삽입
    sys.modules["firecrawl"] = mock_module
    yield mock_module, mock_client  # 테스트에서 필요할 경우 접근할 수 있도록 yield

    # 테스트 후 정리
    if "firecrawl" in sys.modules:
        del sys.modules["firecrawl"]


class TestFireCrawlLoader:
    """Test FireCrawlLoader."""

    def test_load_extract_mode(
        self, mock_firecrawl: Tuple[MagicMock, MagicMock]
    ) -> List[Document]:
        """Test loading in extract mode."""
        # fixture에서 모킹된 객체 가져오기
        _, mock_client = mock_firecrawl

        params = {
            "prompt": "extract the title and main contents(write your own prompt here)",
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "main contents": {"type": "string"},
                },
                "required": ["title", "main contents"],
            },
            "enableWebSearch": False,
            "ignoreSitemap": False,
            "showSources": False,
            "scrapeOptions": {
                "formats": ["markdown"],
                "onlyMainContent": True,
                "headers": {},
                "waitFor": 0,
                "mobile": False,
                "skipTlsVerification": False,
                "timeout": 30000,
                "removeBase64Images": True,
                "blockAds": True,
                "proxy": "basic",
            },
        }

        # FireCrawlLoader 인스턴스 생성 및 실행
        loader = FireCrawlLoader(
            url="https://example.com", api_key="fake-key", mode="extract", params=params
        )
        docs = list(loader.lazy_load())  # lazy_load 메서드 호출

        # 검증
        assert len(docs) == 1
        assert isinstance(docs[0].page_content, str)

        # extract 메서드가 올바른 인자로 호출되었는지 확인
        mock_client.extract.assert_called_once_with(
            ["https://example.com"], params=params
        )

        # 응답이 문자열로 변환되었으므로 각 속성이 문자열에 포함되어 있는지 확인
        assert "extracted title" in docs[0].page_content
        assert "extracted main contents" in docs[0].page_content
        assert "success" in docs[0].page_content

        return docs
