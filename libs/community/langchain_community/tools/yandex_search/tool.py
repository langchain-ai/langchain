"""Tool for the Yandex Search API."""

from typing import Dict, List, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.yandex_search import YandexSearchAPIWrapper


class YandexSearchInput(BaseModel):
    """Поисковый запрос для поисковой системы."""

    query: str = Field(description="Текст запроса")


class YandexSearchResult(BaseModel):
    """Найденный документ."""

    url: Optional[int] = Field(default=None, description="URL страницы документа")
    content: Optional[str] = Field(default=None, description="Текст документа")


class YandexSearchOutput(BaseModel):
    """Все результаты поиска в поисковой системе."""

    results: List[YandexSearchResult] = Field(description="Результаты поиска")


class YandexSearchResults(BaseTool):
    name: str = "yandex_search_results_json"
    description: str = (
        "Поисковая система, оптимизированная для получения всесторонних, "
        "точных и надежных результатов. "
        "Полезна, когда нужно ответить на вопросы о текущих событиях. "
        "Ввод должен быть поисковым запросом."
    )
    api_wrapper: YandexSearchAPIWrapper = Field(default_factory=YandexSearchAPIWrapper)  # type: ignore[arg-type]
    args_schema: Type[BaseModel] = YandexSearchInput
    max_results: int = 10
    return_schema: Type[BaseModel] = YandexSearchOutput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool."""

        results = self.api_wrapper.results(query)
        return results[: self.max_results]

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[List[Dict], str]:
        """Use the tool asynchronously."""

        results = await self.api_wrapper.results_async(query)
        return results[: self.max_results]
