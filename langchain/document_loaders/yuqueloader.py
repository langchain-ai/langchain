"""
    -- @Time    : 2023/4/25 12:39
    -- @Author  : yazhui Yu
    -- @email   : yuyazhui@bangdao-tech.com
    -- @File    : yuqueloader
    -- @Software: Pycharm
"""
from typing import List, Tuple

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class YUQUELoader(BaseLoader):
    """Loader that loads yuque transcripts."""

    def __init__(self, custom_space: str = 'www', user_agent: str = '', user_token: str = '', **kwargs):
        """
        Initialize with yuque information.
        :param
            custom_space: 访问空间内资源的 API 需要使用空间对应的域名,
                          参考 https://www.yuque.com/yuque/developer/api 中的 Overview 基本路径
            user_agent: 必传参数, 可以任意，default: test
            user_token: 必传参数, 语雀账户的 token
            kwargs:
                    暂时只设置四个额外参数
                    knowledge_base_id: str or List[str] 知识库的 id , default: [],
                                        example: '37316289'

                    knowledge_base_name: str or List[str] 知识库的 name , default: [],
                                        example: '大数据分析模型建设'

                    docs_slug: str or List[str] 文档 slug, default: [],
                                        example: ['li93ug0ghx3eagsm', 'apxbq1pm8gsai4y6']
                                        Tips: 文档 slug 取值: 一般为文档链接 `/` 分割的最后一个字符串
                                        example:
                                            下面文档的对应的 slug 为 islb9vmfg74lbrus
                                            https://bdywzjj.yuque.com/fg16av/ui17dl/islb9vmfg74lbrus

                    docs_title: str or List[str] 文档 title, default: [], 如果标题有符号，建议以 docs_slug 传入.

                    1.传入 docs_slug 和 docs_title 任意多个参数时, 则只下载 docs_slug
                      和 docs_title 中所有的文档;

                    2.在传入 docs_slug(docs_title) 时, 还传 knowledge_base_id 或
                      knowledge_base_name 时, 则必须保证 docs_slug(docs_title) 中所有的文档
                      在 knowledge_base_id 或 knowledge_base_name 的知识库下, 否则只下载在
                      knowledge_base_id 或 knowledge_base_name 知识库中的文档;

                    3.在 docs_slug 和 docs_title 缺失时, 传入 knowledge_base_id
                      和 knowledge_base_name 参数中的任意多个参数后, 则会加载制定知识库的所有文档;

                    4.若四个参数均没有传入, 则默认加载 token 权限下的所有知识库的文档。
        """
        self.custom_space = custom_space
        self.user_agent = user_agent
        self.user_token = user_token
        self.kwargs = kwargs

    def load(self) -> List[Document]:
        """Load from yuque docs api."""
        try:
            from langchain.utilities.yuque_api import sync, YuQueDocs
        except ImportError:
            raise ValueError(
                "requests package not found, please install it."
            )

        yq = YuQueDocs(custom_space=self.custom_space,
                       user_agent=self.user_agent,
                       user_token=self.user_token,
                       **self.kwargs)
        yq_docs = sync(yq.load_docs())

        results = []
        for yq_doc in yq_docs:
            page_content, metadata = self._get_metadata_and_page_content(yq_doc)
            doc = Document(page_content=page_content, metadata=metadata)
            results.append(doc)

        return results

    @staticmethod
    def _get_metadata_and_page_content(yq_doc: dict) -> Tuple[str, dict]:
        page_content = yq_doc.get("data", "")
        metadata = {
            "knowledge_base_id": yq_doc.get("knowledge_base_id", ""),
            "knowledge_base_name": yq_doc.get("knowledge_base_name", ""),
            "title": yq_doc.get("title", ""),
            "slug": yq_doc.get("slug", ""),
            "description": yq_doc.get("description", ""),
        }
        return page_content, metadata
